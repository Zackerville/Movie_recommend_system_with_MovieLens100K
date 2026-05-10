"""[A] Abstract Problem + RecommendationProblem cho hệ gợi ý phim."""

from abc import ABC, abstractmethod
import numpy as np


class Problem(ABC):
    """
    Lớp trừu tượng bài toán tìm kiếm (AIMA-style).

    Bài toán = (S0, A, T, G, c):
        S0  — trạng thái khởi đầu
        A   — tập hành động áp dụng được trên mỗi trạng thái
        T   — hàm chuyển trạng thái  T(s, a) -> s'
        G   — điều kiện mục tiêu
        c   — hàm chi phí bước
    """

    def __init__(self, initial):
        self.initial = initial

    @abstractmethod
    def actions(self, state):
        """Trả về danh sách hành động có thể từ state."""

    @abstractmethod
    def result(self, state, action):
        """Trả về trạng thái sau khi áp dụng action lên state."""

    @abstractmethod
    def goal_test(self, state):
        """True nếu state là mục tiêu."""

    def path_cost(self, c, state1, action, state2):
        """Chi phí mặc định = 1 mỗi bước."""
        return c + 1

    def value(self, state):
        """Giá trị trạng thái cho local search (maximize)."""
        return 0.0


class RecommendationProblem(Problem):
    """
    Đặc tả formal bài toán tìm kiếm cho hệ gợi ý phim Top-K.

    Không gian trạng thái:
        S0   = ()                         — danh sách trống ban đầu
        S    = tuple of movie_ids         — bộ phim đã chọn (immutable)
        A(s) = {add(m) | m ∈ pool, m ∉ s}
        T(s, add(m)) = s + (m,)
        G    = {s | len(s) == K}
        c(s, add(m), s') = -score(m)     — minimize cost ≡ maximize relevance

    Độ phức tạp:
        Branching factor b ≈ |pool| − depth
        Depth d = K
        BFS/UCS: O(b^K) — cần giới hạn pool size ≤ 50

    Heuristic admissible h(s):
        h(s) = −(Σ top K−|s| remaining scores)
        Admissible: h(s) ≤ h*(s) vì top remaining là upper bound.
        Consistent: h(s) ≤ c(s,a,s') + h(s') (triangle inequality).

    Args:
        candidate_pool : list of movie_ids đã pre-filter.
        scores         : dict {movie_id: float} predicted relevance.
        K              : số phim cần gợi ý.
    """

    def __init__(self, candidate_pool, scores, K=10):
        super().__init__(initial=())
        self.scores = scores
        self.K = K
        # Sắp xếp sẵn theo score giảm dần — dùng cho heuristic + BFS ordering
        self._sorted_cands = sorted(
            candidate_pool,
            key=lambda m: scores.get(m, 0.0),
            reverse=True,
        )
        self._score_arr = np.array(
            [scores.get(m, 0.0) for m in self._sorted_cands], dtype=np.float64
        )

    def actions(self, state):
        """Các phim chưa chọn, sắp xếp theo score giảm dần."""
        selected = set(state)
        return [m for m in self._sorted_cands if m not in selected]

    def result(self, state, action):
        """Thêm phim vào danh sách."""
        return state + (action,)

    def goal_test(self, state):
        """Đã chọn đủ K phim."""
        return len(state) == self.K

    def path_cost(self, c, state1, action, state2):
        """Chi phí bước = âm relevance score của phim được thêm."""
        return c - self.scores.get(action, 0.0)

    def score_of_state(self, state):
        """Tổng relevance score của danh sách hiện tại."""
        return sum(self.scores.get(m, 0.0) for m in state)

    def heuristic(self, state):
        """
        Admissible heuristic h(s) = −(sum top K−|s| remaining scores).

        Chứng minh admissible:
          h*(s) = remaining cost tối ưu = −(sum of best K−|s| movies còn lại)
          h(s) chọn đúng top K−|s| movies → không overestimate → h(s) ≤ h*(s) ✓

        Chứng minh consistent:
          h(s) ≤ c(s,a,s') + h(s')
          Với a là top movie còn lại:
            h(s)  = −score(a) − (top K−|s|−1 excl. a)
            c + h(s') = −score(a) + (−top K−|s|−1 excl. a)
          → h(s) = c + h(s') → consistent ✓
        """
        remaining = self.K - len(state)
        if remaining <= 0:
            return 0.0
        selected = set(state)
        rem_scores = [
            self.scores.get(m, 0.0)
            for m in self._sorted_cands
            if m not in selected
        ]
        return -sum(rem_scores[:remaining])
