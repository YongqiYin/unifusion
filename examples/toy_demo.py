"""End-to-end tour of the controller on a synthetic three-modality corpus.

Everything here is generated in-process: no dataset, no model download, no GPU.
The purpose is to show the full contract (corpus -> score rows -> context set)
and to print *why* a context was assembled the way it was.

Run with::

    python examples/toy_demo.py

The metric values printed at the end come from a synthetic corpus and only
verify that the plumbing is wired correctly. They are not a benchmark.
"""
from __future__ import annotations

import numpy as np

from unifusion import (ControllerConfig, Corpus, CorpusItem, QueryScores,
                       UniFusionController, dense_concat, naive_fusion)
from unifusion.metrics import aggregate_ranking, eval_ranking

RNG = np.random.default_rng(7)
DIM = 64
N_PER_MODALITY = 90

ENTITIES = ["aurora", "basalt", "cinder", "dunes", "ember", "fjord",
            "garnet", "harbor", "indigo", "juniper"]
TOPICS = ["stadium", "bridge", "rail", "festival", "turbine", "canal"]

# Distractor vocabulary: shared across modalities, which is what makes the
# dense channel confuse a table row with a text paragraph.
SHARED = ["annual", "reported", "region", "located", "district", "capacity"]


def build_corpus():
    items = []
    for i in range(N_PER_MODALITY):
        entity = ENTITIES[i % len(ENTITIES)]
        topic = TOPICS[i % len(TOPICS)]
        shared = " ".join(RNG.choice(SHARED, size=3, replace=False))
        # every text starts with "<channel word> <entity>" so that the entity
        # is always the second token, whatever the modality
        items.append(CorpusItem(
            id=f"tat_{i:03d}",
            text=f"table {entity} row {topic} {shared} capacity {1000 + i * 37}",
            modality="table"))
        items.append(CorpusItem(
            id=f"kg_{i:03d}",
            text=f"entity {entity} relation {topic} {shared} linked to {TOPICS[(i + 1) % len(TOPICS)]}",
            modality="kg"))
        items.append(CorpusItem(
            id=f"nq_{i:03d}",
            text=f"passage {entity} describes the {topic} {shared} during construction",
            modality="text"))
    return Corpus(items)


def make_query_set(corpus):
    """Each query targets one gold chunk; a sibling chunk is mildly relevant."""
    gold_indices = RNG.choice(len(corpus), size=60, replace=False)
    queries, gold, qrels = [], [], []
    for gi in gold_indices:
        gold_item = corpus[int(gi)]
        tokens = gold_item.text.split()
        # query = the discriminative tokens of the gold chunk, shuffled
        picked = RNG.choice(tokens[1:], size=4, replace=False).tolist()
        queries.append(" ".join(picked))
        gold.append(gold_item)
        # a same-modality, same-entity chunk: relevant but not the answer
        entity = gold_item.text.split()[1]
        sibling = next(i for i in range(len(corpus))
                       if i != gi and corpus[i].modality == gold_item.modality
                       and corpus[i].text.split()[1] == entity)
        qrels.append({gold_item.id: 2, corpus[sibling].id: 1})
    return queries, gold, qrels


def score_rows(corpus, queries, gold):
    """Fake a dense encoder (noisy) and a lexical retriever (exact)."""
    doc_vectors = RNG.normal(size=(len(corpus), DIM))
    doc_vectors /= np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    dense_rows, sparse_rows, prob_rows = [], [], []
    for query, gold_item in zip(queries, gold):
        noise = RNG.normal(size=DIM)
        # the dense channel is informative but not decisive: it is the mix of
        # channels that has to carry the query
        query_vector = doc_vectors[corpus.index_of[gold_item.id]] + 1.5 * noise / np.linalg.norm(noise)
        query_vector /= np.linalg.norm(query_vector)
        dense_rows.append(doc_vectors @ query_vector)

        query_tokens = set(query.lower().split())
        sparse_rows.append(np.asarray(
            [len(query_tokens & corpus.tokens[j]) / max(1, len(query_tokens))
             for j in range(len(corpus))], dtype=np.float32))

        # the router is confident about the modality the gold chunk lives in,
        # with the neighbouring modalities partly activated
        probs = {m: 0.10 + 0.10 * RNG.random() for m in ("text", "table", "kg")}
        probs[gold_item.modality] = 0.62 + 0.33 * RNG.random()
        prob_rows.append(probs)
    return dense_rows, sparse_rows, prob_rows


def main() -> None:
    corpus = build_corpus()
    queries, gold, qrels = make_query_set(corpus)
    dense_rows, sparse_rows, prob_rows = score_rows(corpus, queries, gold)
    print(f"corpus: {len(corpus)} chunks {corpus.modality_counts()}")
    print(f"queries: {len(queries)}\n")

    controller = UniFusionController(corpus, ControllerConfig(budget=10))
    results = {"dense_concat": [], "naive_fusion": [], "unifusion": []}
    for query, dense, sparse, probs, rel in zip(queries, dense_rows, sparse_rows, prob_rows, qrels):
        dense_ids = [corpus.ids[i] for i in dense_concat(dense, budget=10)]
        naive_ids = [corpus.ids[i] for i in naive_fusion(dense, sparse, budget=10, recall_k=30)]
        selection = controller.select(query, QueryScores(dense=dense, sparse=sparse, probs=probs))
        results["dense_concat"].append(eval_ranking(dense_ids, rel))
        results["naive_fusion"].append(eval_ranking(naive_ids, rel))
        results["unifusion"].append(eval_ranking(selection.ids, rel))

    print(f"{'method':14s}{'ndcg@5':>9s}{'map@5':>9s}{'p@5':>9s}{'ndcg@1':>9s}")
    for name, rows in results.items():
        mean = aggregate_ranking(rows)
        print(f"{name:14s}{mean['ndcg@5']:9.3f}{mean['map@5']:9.3f}"
              f"{mean['p@5']:9.3f}{mean['ndcg@1']:9.3f}")

    # ------------------------------------------------------------ one trace
    query, dense, sparse, probs = queries[0], dense_rows[0], sparse_rows[0], prob_rows[0]
    selection = controller.select(query, QueryScores(dense=dense, sparse=sparse, probs=probs))
    weights = ("high-entropy" if selection.entropy >= controller.config.tau_u else "low-entropy")
    print(f"\ntrace for query: {query!r}")
    print(f"  router          : {', '.join(f'{m}={p:.2f}' for m, p in selection.probs.items())}")
    print(f"  entropy u(q)    : {selection.entropy:.3f} -> {weights} weights")
    print(f"  active set A(q) : {selection.active_modalities}")
    print(f"  candidate pool  : {selection.pool_size}")
    print(f"  modality mix    : {selection.modality_mix(corpus)}")
    print("  selected:")
    for rank, index in enumerate(selection.indices, start=1):
        print(f"    {rank:2d}. [{corpus[index].modality:5s}] {corpus[index].id}  "
              f"{corpus[index].text[:58]}")
    print("\nNote: metrics above come from a synthetic corpus and only exercise the code path.")


if __name__ == "__main__":
    main()
