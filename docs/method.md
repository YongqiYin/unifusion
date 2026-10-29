# Method notes: Algorithm 1 in code

This note maps every step of the assembler to the function that implements it,
and records the decisions that are easy to get wrong.

## Notation

| Symbol | Meaning | Where |
|---|---|---|
| `M = {text, table, kg}` | modality vocabulary | `unifusion/types.py::MODALITIES` |
| `p_m(q)` | router probability that modality `m` is needed | `ModalityRouter.predict_proba` |
| `u(q)` | normalised predictive entropy | `entropy.normalized_entropy` |
| `A(q)` | active modality set | `entropy.active_modalities` |
| `B` | context budget | `ControllerConfig.budget` |
| `K_pool` | size of the merged candidate pool | `ControllerConfig.pool_cap` |
| `s_uni(d)` | fused retrieval score | `UniFusionController._rerank` |
| `g(d | S)` | marginal gain of adding `d` to `S` | `UniFusionController._pack` |

## Stage 1 — per-modality recall

`UniFusionController._build_pool`

Each modality has its own dense and sparse channel. For modality `m` the
controller takes the top `K_d` dense and the top `K_s` sparse candidates among
the rows whose modality is `m`, unions the `2M` lists and de-duplicates. If the
union exceeds `K_pool`, candidates are ranked by an even mix of both scores.

Two properties matter here:

* the channels are never pooled across modalities before this step, which is
  what keeps a large table block from crowding out a small graph block;
* the pool cap is applied after the union, so a dominant modality cannot
  consume the whole pool.

## Stage 2 — entropy-conditioned rerank

`UniFusionController._rerank`

```
s_uni(d) = w_d · ŝ_dense(d) + w_s · ŝ_sparse(d) + α · ovl(q, d) + γ · p_{m(d)}(q)
```

`ŝ` denotes min-max normalisation *inside the pool* — normalising over the full
corpus would compress every candidate into a narrow band and make the channel
weights meaningless. `ovl(q, d)` is the fraction of query tokens present in the
chunk, in `[0, 1]`.

The weight pair is the only place where `u(q)` acts:

| `u(q)` | `(w_d, w_s)` | reading |
|---|---|---|
| `< τ_u` | `(0.76, 0.24)` | the router knows which modality is needed; trust dense |
| `≥ τ_u` | `(0.68, 0.32)` | the query mixes modalities; lexical matching is safer |

Setting `use_entropy_adaptation = False` pins the pair to the low-entropy
branch, which is exactly the "no entropy adaptation" ablation.

## Stage 3 — budgeted packing

`UniFusionController._pack`

1. **Anchors.** The top `n_d` candidates of the dense channel and the top `n_r`
   of the fused ordering enter first. If every anchor happens to come from the
   dense channel, the highest-ranked non-dense candidate is promoted, so the
   fused ordering always contributes at least one chunk.
2. **Quotas.** `Q_m = ⌈B · p_m / Σ_{m' ∈ A} p_{m'}⌉` over the active set. The
   ceilings can overshoot `B`; that is intentional and harmless because the
   quotas are *soft*.
3. **Greedy fill.** At each step the candidate with the largest marginal gain
   is taken, provided its modality still has residual quota. When every
   remaining candidate is blocked, the quotas are released and the loop
   continues, so a budget is never left partially filled because of an
   exhausted quota.
4. **Presentation order.** Survivors are re-sorted by `s_uni`, which is the
   order the reader sees.

The redundancy term is recomputed against the *current* set `S` on every step,
which makes the criterion submodular but not additive: two near-duplicate
chunks cannot both be "cheap" at the same time.

## Invariants

These hold for every input and are covered by `tests/test_controller.py`:

* `len(context) ≤ min(budget, pool size)`;
* no chunk appears twice;
* the result is deterministic for identical inputs (stable sorts throughout);
* a corpus with an empty modality still yields a full context;
* constant (all-zero) score rows do not crash the packer.

## Failure modes worth knowing

| Symptom | Cause |
|---|---|
| context collapses to one modality | `tau_a` too high, so `A(q)` excludes the alternatives and the quota denominator is a single modality |
| no entropy effect at all | router outputs near 0/1, so `u(q)` never crosses `tau_u` |
| fused order equals dense order | `alpha` and `gamma` are small relative to `w_d`; normalisation of `ŝ` was skipped |
| empty answers from the reader | the generation budget was consumed by a reasoning scratchpad — see `llm.reader.strip_reasoning` |
