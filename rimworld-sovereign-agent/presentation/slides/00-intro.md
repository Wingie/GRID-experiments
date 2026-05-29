# Sovereign Game Agents

teaching a 1.5B-parameter SLM to play **RimWorld**, **EVE**, **Pokémon Red**, and *The Legend of Zelda: The Minish Cap*

— at the embedding layer.

---

## Why "sovereign"?

the model doesn't read the manual every turn.

it learns the game's **vocabulary**, **entities**, and **workflows** *into the weights*, so inference can spend its context budget on the situation in front of it.

— extended tokenizer + dual semantic IDs + mmproj vision + discrete diffusion + self-play. all on a single RTX 3090.

---

## Why this audience can heckle the demo

the agent has a parallel **commentary channel**. while it plays, you can ask questions, and it answers grounded in the same SIDs that drive its play.

it never asks you back.
