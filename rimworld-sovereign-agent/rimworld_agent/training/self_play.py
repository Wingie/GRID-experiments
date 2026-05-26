"""Self-play training loop (project spec §7c).

Bootstrap with scripted/random play, then iterate: play K episodes with the current model,
keep the top-50% by total reward (GRPO-style filtering — the game's reward *is* the reward
model, no separate critic), and fine-tune on the good episodes with a discrete-diffusion
objective over the action tokens (mask 30–70% of action tokens, denoise to recover the full
sequence). The model's own good play reinforces itself; bad play is filtered out.

GPU + live game only; written-but-unverified here. The :class:`ModelPolicy` and the masking
objective are import-safe (torch imported lazily).
"""

from __future__ import annotations

import random
from pathlib import Path

from rimworld_agent.game.action_space import MAX_ACTIONS_PER_TURN, Action, action_names, parse_actions
from rimworld_agent.game.episode_recorder import EpisodeRecorder
from rimworld_agent.game.game_loop import Observation, play_episode
from rimworld_agent.utils import cfg_get, ensure_dir, get_logger, write_json

log = get_logger("self_play")


def _sid_lookups(cfg):
    """Build (rsids_for, wsids_for) callables from the precomputed SID assignments.

    ``rsids_for(def_names)`` -> the RSID inline strings of those entities (perception);
    ``wsids_for(def_names)`` -> the WSID inline strings (workflow), skipping entities that
    have no WSID yet. Returns (None, None) if no assignments file exists.
    """
    import json
    from pathlib import Path

    path = Path(cfg_get(cfg, "paths.sid_assignments_file", "results/sid_assignments.json"))
    if not path.exists():
        return None, None
    assigns = json.loads(path.read_text())["assignments"]
    rmap = {dn: a["rsid_inline"] for dn, a in assigns.items() if a.get("rsid_inline")}
    wmap = {dn: a["wsid_inline"] for dn, a in assigns.items() if a.get("wsid_inline")}

    def rsids_for(def_names):
        return [rmap[n] for n in def_names if n in rmap]

    def wsids_for(def_names):
        return [wmap[n] for n in def_names if n in wmap]

    return rsids_for, wsids_for


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
class RandomPolicy:
    """Iteration-0 bootstrap: emit 1–5 random valid actions (no params resolved)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.names = [n for n in action_names() if n not in ("select_at", "drag_select")]

    def __call__(self, obs: Observation) -> tuple[str, list[Action]]:
        k = self.rng.randint(1, MAX_ACTIONS_PER_TURN)
        actions = [Action(action=self.rng.choice(self.names), reason="random bootstrap") for _ in range(k)]
        return "random bootstrap", actions


class ModelPolicy:
    """Wrap the trained SLM (+ mmproj) as a :class:`game_loop.Policy`."""

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.state_encoder = None
        if cfg_get(cfg, "self_play.use_vision", True):
            from rimworld_agent.vision.state_encoder import StateEncoder

            self.state_encoder = StateEncoder(cfg, cfg_get(cfg, "paths.mmproj_ckpt", None))

    def _build_prompt(self, obs: Observation) -> str:
        from rimworld_agent.training.prepare_data import render_game_state

        return (
            "You are a RimWorld colony manager. Analyse the situation and provide up to 5 "
            "actions with reasoning. Use RSIDs when describing what you observe, WSIDs when "
            "planning actions.\n"
            + render_game_state(obs.game_state, obs.visible_rsids)
            + "\n<REASONING>"
        )

    def __call__(self, obs: Observation) -> tuple[str, list[Action]]:
        import torch

        prompt = self._build_prompt(obs)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # (Visual tokens from self.state_encoder.encode(obs.screenshots) would be spliced
        # into inputs_embeds here when vision is enabled.)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg_get(self.cfg, "self_play.max_new_tokens", 512),
                do_sample=True,
                temperature=cfg_get(self.cfg, "self_play.temperature", 0.8),
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=False)
        actions = parse_actions(text)[:MAX_ACTIONS_PER_TURN]
        reasoning = text.split("<REASONING>", 1)[-1].split("<ACTION_START>", 1)[0].strip()
        return reasoning, actions


# --------------------------------------------------------------------------- #
# Discrete-diffusion action denoising objective
# --------------------------------------------------------------------------- #
def mask_action_tokens(input_ids, action_token_ids: set[int], tokenizer, mask_rate_range=(0.3, 0.7), rng=None):
    """Randomly replace a fraction of *action* tokens with <mask>; return (corrupted, labels).

    Only action-region tokens are corrupted; everything else is context the model conditions
    on. Labels are -100 except at masked positions (standard denoising target).
    """
    import torch

    rng = rng or random
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else tokenizer.eos_token_id
    corrupted = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    rate = rng.uniform(*mask_rate_range)
    for b in range(input_ids.size(0)):
        positions = [i for i, t in enumerate(input_ids[b].tolist()) if t in action_token_ids]
        rng.shuffle(positions)
        for i in positions[: int(len(positions) * rate)]:
            labels[b, i] = input_ids[b, i]
            corrupted[b, i] = mask_id
    return corrupted, labels


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #
def run_self_play(cfg) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rimworld_agent.game.rimapi_client import RimAPIClient

    client = RimAPIClient(cfg_get(cfg, "game.rimapi_url", "http://127.0.0.1:7860"))
    model_dir = cfg_get(cfg, "paths.model_out_dir", "data/models") + "/qlora"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")

    n_iters = cfg_get(cfg, "self_play.iterations", 5)
    k = cfg_get(cfg, "self_play.episodes_per_iter", 20)
    ep_root = ensure_dir(cfg_get(cfg, "paths.episodes_dir", "data/episodes"))
    rsids_for, wsids_for = _sid_lookups(cfg)
    metrics = []

    for it in range(n_iters):
        policy = RandomPolicy() if it == 0 and cfg_get(cfg, "self_play.bootstrap_random", True) else ModelPolicy(cfg, model, tokenizer)
        episodes = []
        for j in range(k):
            rec = EpisodeRecorder(episode_id=f"it{it:02d}_ep{j:03d}", root=ep_root,
                                  scenario=cfg_get(cfg, "game.scenario", "crashlanded"),
                                  difficulty=cfg_get(cfg, "game.difficulty", "peaceful"))
            play_episode(client, policy, rec, max_steps=cfg_get(cfg, "eval.episode_length", 150),
                         rsids_for=rsids_for, wsids_for=wsids_for)
            episodes.append(rec.episode)

        episodes.sort(key=lambda e: e.total_reward, reverse=True)
        good = episodes[: max(1, len(episodes) // 2)]
        median = episodes[len(episodes) // 2].total_reward if episodes else 0.0

        if cfg_get(cfg, "self_play.train", True):
            log.info("iter %d: median reward=%.2f, training on top %d", it, median, len(good))
            train_on_episodes(cfg, model, tokenizer, good)
        else:
            log.info("iter %d: median reward=%.2f, recording only (no training)", it, median)
        metrics.append({
            "iteration": it,
            "mean_reward": sum(e.total_reward for e in episodes) / len(episodes),
            "median_reward": median,
            "best_reward": episodes[0].total_reward,
        })

    write_json(metrics, Path(cfg_get(cfg, "paths.results_dir", "results")) / "self_play_metrics.json")
    return {"iterations": n_iters, "metrics": metrics}


def train_on_episodes(cfg, model, tokenizer, episodes) -> None:
    """One pass of discrete-diffusion action denoising over the good episodes."""
    import torch

    from rimworld_agent.game.action_space import action_special_tokens
    from rimworld_agent.training.prepare_data import format_episode_replay

    action_token_ids = {
        tokenizer.convert_tokens_to_ids(t)
        for t in action_special_tokens()
        if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id
    }
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg_get(cfg, "self_play.lr", 1e-5))
    model.train()
    texts = [format_episode_replay(s) for e in episodes for s in e.steps]
    bs = cfg_get(cfg, "self_play.batch_size", 4)
    for start in range(0, len(texts), bs):
        enc = tokenizer(texts[start:start + bs], return_tensors="pt", padding=True,
                        truncation=True, max_length=cfg_get(cfg, "training.max_seq_len", 4096)).to(model.device)
        corrupted, labels = mask_action_tokens(enc["input_ids"], action_token_ids, tokenizer)
        out = model(input_ids=corrupted, attention_mask=enc["attention_mask"], labels=labels)
        opt.zero_grad()
        out.loss.backward()
        opt.step()


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        run_self_play(cfg)

    _run()


if __name__ == "__main__":
    main()
