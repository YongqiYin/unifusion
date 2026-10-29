# UniFusion

Entropy-adaptive context assembly for heterogeneous retrieval-augmented generation.

This repository accompanies the paper:

> **UniFusion: Entropy-Adaptive Context Assembly for Heterogeneous RAG**

Heterogeneous RAG pipelines retrieve from several channels at once — text
passages, table rows, knowledge-graph triples — and then have to decide which
few chunks actually go into the prompt. UniFusion makes that decision in three
stages: it predicts which modalities a query needs, uses the uncertainty of that
prediction to re-weight the retrieval channels, and packs the budget with a
marginal-gain criterion that penalises redundancy and respects soft
per-modality quotas.

## What is here

| Module | Role |
|---|---|
| `unifusion/controller.py` | the assembler: recall → rerank → pack (Algorithm 1) |
| `unifusion/entropy.py` | modality probabilities → normalised predictive entropy (Eq. 3) |
| `unifusion/router/` | multi-label classifier `q ↦ p(q)` with training script (Eq. 2) |
| `unifusion/baselines.py` | `dense_concat` and `naive_fusion` reference builders |
| `unifusion/metrics/` | graded NDCG@k, MAP@k, P@k and SQuAD-style EM / token-F1 |
| `unifusion/retrieval/` | dense encoder wrapper, TF-IDF and BM25 scorers |
| `unifusion/llm/` | extraction prompt, generation client, reasoning-trace stripping |

The controller is dataset-agnostic. It never reads a file: you hand it a corpus
and, per query, one dense score row, one sparse score row and one modality
probability vector.

## Method

For a query `q` with router outputs `p_m(q)`:

1. **Recall.** Each modality keeps its own dense and sparse channel. The top
   candidates of every channel are merged and de-duplicated into a pool of at
   most `K_pool` chunks.
2. **Entropy-conditioned rerank.** With
   `u(q) = Σ_m H_b(p_m(q)) / (M ln 2)`,

   ```
   s_uni(d) = w_d · ŝ_dense(d) + w_s · ŝ_sparse(d) + α · ovl(q, d) + γ · p_{m(d)}(q)
   ```

   where `(w_d, w_s) = (0.76, 0.24)` when `u(q) < τ_u` and `(0.68, 0.32)`
   otherwise: an ambiguous query leans more on the lexical channel, a
   confident one on the dense channel.
3. **Budgeted packing.** Anchors from each channel, then greedy selection by

   ```
   g(d | S) = w1 · π_d + w2 · ovl(q, d) + w3 · p_{m(d)}(q) − λ · max_{d' ∈ S} Jaccard(d, d')
   ```

   with `π_d = 1 / (rank_d + 1)` and soft per-modality quotas
   `Q_m = ⌈B · p_m / Σ_{m' ∈ A} p_{m'}⌉` over the active set
   `A = {m : p_m ≥ τ_a}`.

All symbols and their defaults are in `ControllerConfig`; each ablation switch
(`use_entropy_adaptation`, `use_soft_quotas`, `use_redundancy`, `use_anchors`)
disables exactly one term so a single config expresses every variant.

## Install

```bash
pip install -e .                 # numpy only
pip install -e ".[sparse]"       # + TF-IDF / BM25 scorers
pip install -e ".[dense]"        # + bi-encoder scoring
pip install -e ".[router]"       # + router training
pip install -e ".[llm]"          # + generation client
```

## Quickstart

```python
import numpy as np
from unifusion import ControllerConfig, Corpus, CorpusItem, QueryScores, UniFusionController

corpus = Corpus([
    CorpusItem(id="t1", text="stadium row 1990 capacity", modality="table"),
    CorpusItem(id="x1", text="the stadium was built in 1990", modality="text"),
    CorpusItem(id="g1", text="entity stadium located in the district", modality="kg"),
])

controller = UniFusionController(corpus, ControllerConfig(budget=2))
selection = controller.select(
    "which stadium opened in 1990",
    QueryScores(
        dense=np.array([0.91, 0.72, 0.15]),
        sparse=np.array([0.30, 0.88, 0.05]),
        probs=np.array([0.81, 0.66, 0.12]),   # router: text + table, no graph
    ),
)
print(selection.ids)                 # ordered context
print(selection.entropy)             # u(q), decides the fusion weights
print(selection.modality_mix(corpus))
```

Run the end-to-end tour (synthetic corpus, no downloads, no GPU):

```bash
python examples/toy_demo.py
```

It prints the assembled context of one query together with the router output,
the entropy, the activated modalities, the quota split and the candidate pool
size, so every decision of the controller is inspectable.

## Training the router

```bash
python -m unifusion.router.train \
    --train data/router_train.jsonl --dev data/router_dev.jsonl \
    --backbone microsoft/deberta-base --epochs 3 --batch-size 16 --lr 2e-5 \
    --out runs/router
```

Each JSONL line is `{"query": "...", "modalities": ["text", "table"]}`. Labels
are multi-hot, trained with per-label sigmoid cross-entropy.

## Using your own data

```python
from unifusion.retrieval import DenseEncoder, bm25_scores

encoder = DenseEncoder()                      # any HuggingFace bi-encoder
corpus_vectors = encoder.encode(corpus.texts)
dense = encoder.scores(encoder.encode([query]), corpus_vectors)[0]
sparse = bm25_scores([query], corpus.texts)[0]
probs = router.predict_proba([query])[0]      # or any calibrated estimate
```

## Scope

This repository contains the method implementation. The evaluation corpora, the
trained router checkpoint and the precomputed retrieval scores used in the paper
are not distributed here, so the numbers in the paper cannot be regenerated by
cloning this repository — bring your own corpus, your own retriever stack and
your own relevance judgements, then evaluate with `unifusion.metrics`.

## Tests

```bash
pytest
```

## Citation

```bibtex
@misc{yin_unifusion,
  title  = {UniFusion: Entropy-Adaptive Context Assembly for Heterogeneous RAG},
  author = {Yin, Yongqi},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
