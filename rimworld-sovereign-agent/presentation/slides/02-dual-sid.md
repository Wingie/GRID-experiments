## Two codebooks, not one

motivated by Spotify (NeurIPS 2025): **search-tuned** and **rec-tuned** semantic IDs degrade each other when sharing a codebook.

| | what it captures | trains on |
|---|---|---|
| **READ** `<RSID_L*_*>` | taxonomy — *what IS this?* | structural views (def + label + C# + wiki) |
| **WRITE** `<WSID_L*_*>` | workflow — *what is it USED WITH?* | co-occurrence mined from gameplay (PPMI + SVD) |

`L=3 × K=64` per family — 192 + 192 per-level tokens, < 0.5 % of a 151 k vocab.

---

## Side by side

| Entity | READ | WRITE |
|--------|------|-------|
| SolarGenerator | `Buildings→Power→Solar` | `PowerSetup→Generation→Solar` |
| Battery | `Buildings→Power→Storage` | `PowerSetup→Storage→Battery` |
| ResearchBench | `Buildings→Production→Research` | `PowerSetup→Prerequisites→Bench` |

```
<REASONING> I see <RSID_L3_02> (a bed) ... </REASONING>
<ACTIONS>   <ACTION_1>order_build(Bed, 15, 22) — <WSID_L1_02>: bed for quest</ACTION_1>
            ...
</ACTIONS>
```

---

## The hard training order

READ trains from Defs alone.
WRITE needs **bootstrap gameplay first** — no episodes, no co-occurrence, no WSIDs.

1. READ RQ-VAE
2. RSID-only QLoRA pre-train
3. bootstrap 50–100 episodes
4. mine co-occurrence → WRITE RQ-VAE
5. dual retrain (RSID + WSID)
6. self-play

a tracked failure mode: **SID leakage** — RSIDs in actions or WSIDs in reasoning. `eval_planning` reports it.
