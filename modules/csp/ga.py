"""[B] Genetic Algorithm cho tối ưu hóa danh sách gợi ý Top-K."""

import numpy as np
import random

from .fitness import compute_fitness, batch_fitness


class GeneticAlgorithm:
    """
    GA tối ưu hóa danh sách K phim gợi ý.

    Chromosome : list of K unique movie_ids từ candidate pool
    Fitness    : relevance + genre_diversity − watch_penalty
    Selection  : Tournament selection (size=tournament_size)
    Crossover  : Order Crossover (OX) — giữ thứ tự, tránh trùng lặp
    Mutation   : Random replacement — thay 1 phim bằng phim khác từ pool
    Elitism    : Giữ top-2 cá thể tốt nhất qua mỗi thế hệ

    Args:
        candidate_pool    : list of movie_ids ứng viên (pre-filtered)
        scores            : dict {movie_id: float} predicted relevance
        movie_genre_matrix: np.ndarray (N_MOVIES, N_GENRES)
        watched_ids       : set of movie_ids user đã xem
        K                 : số phim cần gợi ý
        pop_size          : kích thước quần thể
        n_generations     : số thế hệ
        mutation_rate     : xác suất đột biến mỗi chromosome
        crossover_rate    : xác suất lai ghép
        tournament_size   : số cá thể trong mỗi tournament
        random_state      : seed
    """

    def __init__(
        self,
        candidate_pool,
        scores,
        movie_genre_matrix,
        watched_ids,
        K=10,
        pop_size=150,
        n_generations=300,
        mutation_rate=0.15,
        crossover_rate=0.85,
        tournament_size=4,
        random_state=42,
    ):
        self.candidate_pool    = list(candidate_pool)
        self.scores            = scores
        self.movie_genre_matrix = movie_genre_matrix
        self.watched_ids       = set(watched_ids)
        self.K                 = K
        self.pop_size          = pop_size
        self.n_generations     = n_generations
        self.mutation_rate     = mutation_rate
        self.crossover_rate    = crossover_rate
        self.tournament_size   = tournament_size

        self._rng = np.random.RandomState(random_state)
        random.seed(random_state)

        self.best_solution  = None
        self.best_fitness   = -np.inf
        self.history        = []   # best fitness per generation
        self.mean_history   = []   # mean fitness per generation

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_population(self):
        k = min(self.K, len(self.candidate_pool))
        pop = []
        for _ in range(self.pop_size):
            chrom = list(self._rng.choice(self.candidate_pool, size=k, replace=False))
            pop.append(chrom)
        return pop

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _fitness(self, chrom):
        return compute_fitness(
            chrom, self.scores, self.movie_genre_matrix, self.watched_ids
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population, fitnesses):
        idx = self._rng.choice(len(population), self.tournament_size, replace=False)
        best_i = idx[np.argmax([fitnesses[i] for i in idx])]
        return population[best_i][:]

    # ── Crossover ─────────────────────────────────────────────────────────────

    def _order_crossover(self, p1, p2):
        """
        Order Crossover (OX):
          1. Chọn đoạn [a, b) ngẫu nhiên từ p1 → copy vào child
          2. Điền phần còn lại theo thứ tự của p2, bỏ phần tử đã có
        """
        if random.random() > self.crossover_rate:
            return p1[:], p2[:]

        K = len(p1)
        a = random.randint(0, K - 2)
        b = random.randint(a + 1, K)

        def _ox(par1, par2):
            child = [None] * K
            child[a:b] = par1[a:b]
            used  = set(child[a:b])
            fill  = [x for x in par2 if x not in used]
            ptr   = 0
            for i in range(K):
                if child[i] is None:
                    child[i] = fill[ptr]
                    ptr += 1
            return child

        return _ox(p1, p2), _ox(p2, p1)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def _mutate(self, chrom):
        """Thay ngẫu nhiên 1 phim bằng phim khác từ pool."""
        if random.random() > self.mutation_rate:
            return chrom
        chrom    = chrom[:]
        used     = set(chrom)
        alts     = [m for m in self.candidate_pool if m not in used]
        if not alts:
            return chrom
        pos        = random.randrange(len(chrom))
        chrom[pos] = random.choice(alts)
        return chrom

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, verbose=False):
        """
        Chạy GA và trả về (best_solution, best_fitness).
        Kết quả convergence lưu trong self.history.
        """
        population = self._init_population()
        no_improve = 0

        for gen in range(self.n_generations):
            fitnesses = batch_fitness(
                population, self.scores, self.movie_genre_matrix, self.watched_ids
            )

            # Cập nhật best
            best_idx = int(np.argmax(fitnesses))
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness  = float(fitnesses[best_idx])
                self.best_solution = population[best_idx][:]
                no_improve = 0
            else:
                no_improve += 1

            self.history.append(float(fitnesses.max()))
            self.mean_history.append(float(fitnesses.mean()))

            if verbose and gen % 50 == 0:
                print(f'  Gen {gen:>3}: best={self.best_fitness:.4f}  '
                      f'mean={fitnesses.mean():.4f}')

            # Early stopping nếu không cải thiện 80 generation
            if no_improve >= 80:
                break

            # Elitism: giữ top-3
            elite_idx = np.argsort(fitnesses)[-3:]
            new_pop   = [population[i][:] for i in elite_idx]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._order_crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            population = new_pop

        return self.best_solution, self.best_fitness
