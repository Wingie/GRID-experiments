## The recipe

| Layer | What it does |
|------|--------------|
| **Extended tokenizer** | mined game vocab + dual SID tokens + action + vision tokens |
| **Dual semantic IDs**  | RQ-VAE for entity *taxonomy* (READ) and *workflow* (WRITE) |
| **mmproj vision**      | SigLIP → MLP into LLM embedding space, aligned to RSIDs |
| **Discrete diffusion** | parallel action generation + iterative denoising |
| **Self-play**          | GRPO-style filtering; the game *is* the reward model |

---

## RimWorld is the testbed, not the goal

the same pipeline applies to:

- testing your web app
- navigating your IDE
- automating your internal tools

anywhere with a built-in reward signal.

---

## Built on `vocab-extend-qlora`

the tokenizer-extension / RQ-VAE / QLoRA / GGUF-export machinery is **reused as a dependency** — `import src.*`.

the new package is `rimworld_agent` so `src` unambiguously resolves to the reused project.
