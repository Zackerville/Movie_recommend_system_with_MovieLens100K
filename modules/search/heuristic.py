"""[A] Admissible heuristic cho A* Top-K recommendation."""


def h_top_remaining(state, sorted_cands, scores, K):
    """
    h(s) = −(sum of top K−|s| remaining scores).

    ADMISSIBILITY PROOF
    ───────────────────
    Định nghĩa:
      h*(s) = chi phí tối ưu còn lại để đạt goal từ s
            = −(max sum của K−|s| phim còn lại có thể chọn)

    h(s) lấy top K−|s| phim theo score giảm dần từ pool còn lại.
    Bất kỳ completion nào cũng không thể có tổng score cao hơn top K−|s|.
    → h(s) ≤ h*(s)  ⟹  ADMISSIBLE ✓

    CONSISTENCY PROOF
    ─────────────────
    Cần chứng minh: h(s) ≤ c(s, a, s') + h(s')
      c(s, a, s') = −score(a)
      h(s)  = −[score(a) + top K−|s|−1 scores excl. a]   (vì a ∈ top K−|s|)
      h(s') = −[top K−|s|−1 scores excl. a]
    ⟹ h(s) = c(s, a, s') + h(s')  (equality)  ⟹  CONSISTENT ✓

    Args:
        state        : tuple of currently selected movie_ids.
        sorted_cands : list of movie_ids sorted by descending score.
        scores       : dict {movie_id: float}.
        K            : target list size.

    Returns:
        float heuristic value (≤ 0).
    """
    remaining = K - len(state)
    if remaining <= 0:
        return 0.0
    selected = set(state)
    rem_scores = [
        scores.get(m, 0.0) for m in sorted_cands if m not in selected
    ]
    return -sum(rem_scores[:remaining])
