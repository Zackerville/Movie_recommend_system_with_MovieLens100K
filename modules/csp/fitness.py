"""[B] Fitness function cho GA recommendation list optimization."""

import numpy as np

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]
N_GENRES = len(GENRE_COLS)


def compute_fitness(chromosome, scores, movie_genre_matrix, watched_ids,
                    alpha=1.0, beta=0.35, gamma=0.5):
    """
    Fitness = alpha * relevance + beta * genre_diversity - gamma * watch_penalty

    relevance      : mean normalized predicted score của K phim trong chromosome
    genre_diversity: số thể loại phân biệt được cover / N_GENRES
    watch_penalty  : tỉ lệ phim đã xem trong chromosome

    Args:
        chromosome        : list of K movie_ids
        scores            : dict {movie_id: float} predicted relevance
        movie_genre_matrix: np.ndarray (N_MOVIES, N_GENRES) genre flags
        watched_ids       : set of movie_ids user đã xem
        alpha, beta, gamma: trọng số các thành phần

    Returns:
        float fitness value (higher = better)
    """
    if not chromosome:
        return 0.0

    K = len(chromosome)

    # ── Relevance ────────────────────────────────────────────────────────────
    raw = np.array([scores.get(m, 0.0) for m in chromosome], dtype=np.float64)
    s_min, s_max = raw.min(), raw.max()
    if s_max > s_min:
        relevance = float((raw - s_min).mean() / (s_max - s_min))
    else:
        relevance = 1.0

    # ── Genre diversity ───────────────────────────────────────────────────────
    covered = np.zeros(N_GENRES, dtype=bool)
    for movie_id in chromosome:
        idx = movie_id - 1
        if 0 <= idx < len(movie_genre_matrix):
            covered |= movie_genre_matrix[idx].astype(bool)
    diversity = float(covered.sum()) / N_GENRES

    # ── Watch penalty ─────────────────────────────────────────────────────────
    watch_penalty = sum(1 for m in chromosome if m in watched_ids) / K

    return alpha * relevance + beta * diversity - gamma * watch_penalty


def batch_fitness(population, scores, movie_genre_matrix, watched_ids,
                  alpha=1.0, beta=0.35, gamma=0.5):
    """Tính fitness cho toàn bộ population. Trả về np.array."""
    return np.array([
        compute_fitness(chrom, scores, movie_genre_matrix, watched_ids,
                        alpha, beta, gamma)
        for chrom in population
    ], dtype=np.float64)
