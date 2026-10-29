"""Entropy-adaptive context assembly (Algorithm 1).

The controller turns retrieval evidence into an ordered context set of at most
``budget`` chunks in three stages:

======================  ==================================================
Stage 1  (recall)       per-modality dense and sparse channels, merged and
                        de-duplicated into a candidate pool of size K_pool
Stage 2  (rerank)       s_uni(d) = w_d * dense_hat + w_s * sparse_hat
                                   + alpha * ovl(q, d) + gamma * p_{m(d)}(q)
                        with (w_d, w_s) chosen by the query entropy u(q)
Stage 3  (pack)         anchors, then greedy marginal-gain selection under
                        soft per-modality quotas and a redundancy penalty
======================  ==================================================

Both ``w`` pairs and the entropy switch are evaluated per query, so a
modality-ambiguous query (high u) leans more on the sparse channel than a
modality-specific one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .entropy import active_modalities, normalized_entropy
from .types import MODALITIES, ContextSelection, Corpus, QueryScores
from .utils import jaccard, minmax, tokenize, topk_indices


@dataclass
class ControllerConfig:
    """Hyper-parameters of Algorithm 1.

    Defaults are the values used in the paper. Every ablation switch maps to a
    term of the objective, which keeps ablations reproducible from one config.
    """

    budget: int = 10                 # B, chunks in the assembled context
    # --- stage 1 ---
    pool_cap: int = 80               # K_pool
    per_modality_dense_k: int = 20   # K_d per modality channel
    per_modality_sparse_k: int = 20  # K_s per modality channel
    # --- stage 2 ---
    tau_a: float = 0.40              # activation threshold for A(q)
    tau_u: float = 0.75              # entropy switch
    alpha_overlap: float = 0.08      # alpha
    gamma_router: float = 0.10       # gamma
    w_dense_low_entropy: float = 0.76
    w_sparse_low_entropy: float = 0.24
    w_dense_high_entropy: float = 0.68
    w_sparse_high_entropy: float = 0.32
    # --- stage 3 ---
    gain_rank: float = 0.42          # w1, rank prior
    gain_overlap: float = 0.36       # w2, query overlap
    gain_modality: float = 0.22      # w3, modality confidence
    redundancy_lambda: float = 0.08  # lambda
    anchor_dense_k: int = 4          # n_d
    anchor_fused_k: int = 1          # n_r
    # --- ablation switches ---
    use_redundancy: bool = True
    use_entropy_adaptation: bool = True
    use_soft_quotas: bool = True
    use_anchors: bool = True

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.pool_cap < self.budget:
            raise ValueError("pool_cap must be >= budget")
        for name in ("tau_a", "tau_u", "alpha_overlap", "gamma_router",
                     "redundancy_lambda"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")


class UniFusionController:
    """Assembles a context set for one query at a time."""

    def __init__(self, corpus: Corpus, config: ControllerConfig | None = None) -> None:
        self.corpus = corpus
        self.config = config or ControllerConfig()
        self._modality_index = {m: corpus.modality_indices(m) for m in MODALITIES}

    # ------------------------------------------------------------------ API
    def select(self, query: str, scores: QueryScores) -> ContextSelection:
        """Run all three stages for a single query."""
        scores.validate(len(self.corpus))
        probs = scores.as_prob_dict()
        entropy = normalized_entropy(probs)

        pool = self._build_pool(scores.dense, scores.sparse)
        fused, order, dense_norm = self._rerank(
            pool, scores.dense, scores.sparse, query, probs, entropy)
        chosen = self._pack(order, pool, dense_norm, fused, query, probs)

        return ContextSelection(
            indices=chosen,
            ids=[self.corpus.ids[i] for i in chosen],
            entropy=entropy,
            probs=probs,
            active_modalities=active_modalities(probs, self.config.tau_a),
            pool_size=int(pool.size),
        )

    def select_batch(self, queries, score_rows) -> list:
        """Convenience wrapper: ``score_rows`` must pair with ``queries``."""
        if len(queries) != len(score_rows):
            raise ValueError("queries and score_rows must have equal length")
        return [self.select(q, s) for q, s in zip(queries, score_rows)]

    # ------------------------------------------------------- stage 1: recall
    def _build_pool(self, dense_row: np.ndarray, sparse_row: np.ndarray) -> np.ndarray:
        """MergeDedup over independent per-modality dense / sparse channels."""
        cfg = self.config
        parts = []
        for modality in MODALITIES:
            idx = self._modality_index[modality]
            if idx.size == 0:
                continue
            parts.append(idx[topk_indices(dense_row[idx], min(cfg.per_modality_dense_k, idx.size))])
            parts.append(idx[topk_indices(sparse_row[idx], min(cfg.per_modality_sparse_k, idx.size))])
        if not parts:
            raise ValueError("corpus has no candidate to recall from")

        pool = np.unique(np.concatenate(parts)).astype(np.int64)
        if pool.size > cfg.pool_cap:
            # the cap is applied on an even mix of both channels; scores are
            # re-normalised afterwards, inside the surviving pool
            mix = 0.5 * minmax(dense_row[pool]) + 0.5 * minmax(sparse_row[pool])
            pool = pool[topk_indices(mix, cfg.pool_cap)]
        return pool

    # ------------------------------------------------------- stage 2: rerank
    def _rerank(self, pool, dense_row, sparse_row, query: str, probs: dict,
                entropy: float):
        """Fuse both channels inside the pool.

        Returns ``(fused, order, dense_norm)`` where ``order`` indexes into
        ``pool`` in descending fused score.
        """
        cfg = self.config
        dense_norm = minmax(dense_row[pool])
        sparse_norm = minmax(sparse_row[pool])
        q_tokens = tokenize(query)
        overlap = np.asarray(
            [len(q_tokens & self.corpus.tokens[j]) / max(1, len(q_tokens)) for j in pool],
            dtype=np.float64,
        )
        modality_prior = np.asarray(
            [probs.get(self.corpus[j].modality, 0.0) for j in pool], dtype=np.float64)

        if cfg.use_entropy_adaptation and entropy >= cfg.tau_u:
            w_dense, w_sparse = cfg.w_dense_high_entropy, cfg.w_sparse_high_entropy
        else:
            w_dense, w_sparse = cfg.w_dense_low_entropy, cfg.w_sparse_low_entropy

        fused = (w_dense * dense_norm + w_sparse * sparse_norm
                 + cfg.alpha_overlap * overlap + cfg.gamma_router * modality_prior)
        order = np.argsort(-fused, kind="stable")
        return fused, order, dense_norm

    # --------------------------------------------------------- stage 3: pack
    def _pack(self, order, pool, dense_norm, fused, query: str, probs: dict) -> list:
        cfg = self.config
        ranked = [int(pool[i]) for i in order]
        rank_of = {doc: r for r, doc in enumerate(ranked)}
        dense_order = self._dense_order(pool, dense_norm)
        q_tokens = tokenize(query)

        selected: list = []
        chosen: set = set()

        def take(doc: int) -> bool:
            if doc in chosen:
                return False
            selected.append(int(doc))
            chosen.add(int(doc))
            return True

        # -- anchors: the strongest entries of each channel --
        dense_anchor = set(int(d) for d in dense_order[:cfg.anchor_dense_k])
        if cfg.use_anchors:
            for doc in dense_order[:cfg.anchor_dense_k]:
                if len(selected) >= cfg.budget:
                    break
                take(int(doc))
            for doc in ranked[:cfg.anchor_fused_k]:
                if len(selected) >= cfg.budget:
                    break
                take(doc)
            # the fused channel must contribute at least one non-dense candidate
            if selected and all(doc in dense_anchor for doc in selected):
                for doc in ranked:
                    if doc not in dense_anchor and take(doc):
                        break

        quotas = self._quotas(probs, [self.corpus[j].modality for j in selected])

        def marginal_gain(doc: int) -> float:
            tokens = self.corpus.tokens[doc]
            gain = (cfg.gain_rank / (rank_of[doc] + 1.0)
                    + cfg.gain_overlap * (len(q_tokens & tokens) / max(1, len(q_tokens)) if q_tokens else 0.0)
                    + cfg.gain_modality * probs.get(self.corpus[doc].modality, 0.0))
            if cfg.use_redundancy and selected:
                worst = max(jaccard(tokens, self.corpus.tokens[s]) for s in selected)
                gain -= cfg.redundancy_lambda * worst
            return gain

        remaining = [doc for doc in ranked if doc not in chosen]

        def fill(respect_quotas: bool) -> None:
            while len(selected) < cfg.budget and remaining:
                best_doc, best_gain = None, None
                for doc in remaining:
                    modality = self.corpus[doc].modality
                    if (respect_quotas and cfg.use_soft_quotas
                            and modality in quotas["remaining"]
                            and quotas["remaining"][modality] <= 0):
                        continue
                    gain = marginal_gain(doc)
                    if best_gain is None or gain > best_gain:
                        best_doc, best_gain = doc, gain
                if best_doc is None:
                    return  # every candidate is blocked by an exhausted quota
                remaining.remove(best_doc)
                take(best_doc)
                modality = self.corpus[best_doc].modality
                quotas["used"][modality] = quotas["used"].get(modality, 0) + 1
                if modality in quotas["remaining"]:
                    quotas["remaining"][modality] -= 1

        if cfg.use_soft_quotas:
            fill(respect_quotas=True)
            # quotas are soft: any budget they leave unused is released
            fill(respect_quotas=False)
        else:
            fill(respect_quotas=False)

        # final presentation order follows the fused score; ties keep the
        # packing order (stable sort)
        fused_of = {int(pool[i]): float(fused[i]) for i in range(pool.size)}
        selected.sort(key=lambda d: -fused_of[d])
        return selected[:cfg.budget]

    # ------------------------------------------------------------- helpers
    def _dense_order(self, pool: np.ndarray, dense_norm: np.ndarray) -> np.ndarray:
        return pool[np.argsort(-dense_norm, kind="stable")]

    def _quotas(self, probs: dict, anchored_modalities: list) -> dict:
        """Eq. (6): Q_m = ceil(B * p_m / sum_{m' in A} p_{m'}), minus what anchors took."""
        cfg = self.config
        active = active_modalities(probs, cfg.tau_a)
        mass = sum(max(1e-9, probs.get(m, 0.0)) for m in active)
        total = {m: math.ceil(cfg.budget * max(1e-9, probs.get(m, 0.0)) / mass) for m in active}
        used = {m: 0 for m in MODALITIES}
        for modality in anchored_modalities:
            used[modality] = used.get(modality, 0) + 1
        remaining = {m: total.get(m, 0) - used.get(m, 0) for m in active}
        return {"total": total, "used": used, "remaining": remaining, "active": active}
