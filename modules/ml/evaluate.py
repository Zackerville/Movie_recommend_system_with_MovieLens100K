"""[E] Evaluation metrics: RMSE, MAE, Precision@K, Recall@K, NDCG@K."""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Rating prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_rating_prediction(y_true, y_pred):
    """Trả về dict {RMSE, MAE} với y_pred được clip về [1, 5]."""
    y_pred_c = np.clip(np.asarray(y_pred, dtype=float), 1.0, 5.0)
    return {
        'RMSE': round(rmse(y_true, y_pred_c), 4),
        'MAE':  round(mae(y_true,  y_pred_c), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(recommendations, relevant_items, K):
    """
    Precision@K = |top-K ∩ relevant| / K
    relevant_items: set of movie_ids considered relevant (rating ≥ threshold)
    """
    top_k = recommendations[:K]
    if not top_k:
        return 0.0
    hits = sum(1 for m in top_k if m in relevant_items)
    return hits / K


def recall_at_k(recommendations, relevant_items, K):
    """Recall@K = |top-K ∩ relevant| / |relevant|"""
    if not relevant_items:
        return 0.0
    top_k = recommendations[:K]
    hits  = sum(1 for m in top_k if m in relevant_items)
    return hits / len(relevant_items)


def ndcg_at_k(recommendations, relevant_items, K):
    """
    NDCG@K với binary relevance.
        DCG  = Σ rel_i / log2(i+2)
        IDCG = Σ^min(|rel|,K) 1 / log2(i+2)
    """
    top_k = recommendations[:K]
    dcg   = sum(
        1.0 / np.log2(i + 2)
        for i, m in enumerate(top_k) if m in relevant_items
    )
    n_rel = min(len(relevant_items), K)
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(recommendations_dict, test_ratings,
                              K=10, relevance_threshold=4.0):
    """
    Đánh giá recommendation lists trên toàn bộ users.

    Args:
        recommendations_dict : {user_id: [movie_id, ...]} — top-K lists
        test_ratings         : DataFrame (user_id, movie_id, rating)
        K                    : cutoff
        relevance_threshold  : rating tối thiểu để coi là relevant

    Returns:
        dict với Precision@K, Recall@K, NDCG@K, n_users_evaluated
    """
    relevant = (
        test_ratings[test_ratings['rating'] >= relevance_threshold]
        .groupby('user_id')['movie_id']
        .apply(set)
        .to_dict()
    )

    p_list, r_list, n_list = [], [], []

    for uid, recs in recommendations_dict.items():
        rel = relevant.get(uid, set())
        if not rel:
            continue
        p_list.append(precision_at_k(recs, rel, K))
        r_list.append(recall_at_k(recs, rel, K))
        n_list.append(ndcg_at_k(recs, rel, K))

    n = len(p_list)
    if n == 0:
        return {f'Precision@{K}': 0.0, f'Recall@{K}': 0.0,
                f'NDCG@{K}': 0.0, 'n_users_evaluated': 0}

    return {
        f'Precision@{K}': round(float(np.mean(p_list)), 4),
        f'Recall@{K}':    round(float(np.mean(r_list)), 4),
        f'NDCG@{K}':      round(float(np.mean(n_list)), 4),
        'n_users_evaluated': n,
    }
