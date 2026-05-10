"""[A] BFS, DFS, UCS — uninformed search baselines."""

import time
from collections import deque
import heapq


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stats(nodes_expanded, t0, path_cost):
    return {
        'nodes_expanded': nodes_expanded,
        'time_sec':       round(time.time() - t0, 5),
        'path_cost':      round(path_cost, 4) if path_cost is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BFS
# ─────────────────────────────────────────────────────────────────────────────

def bfs(problem, branch_limit=15):
    """
    Breadth-First Search: level-order traversal.

    Với RecommendationProblem (candidate_pool đã sort by score):
      - Mỗi level = thêm 1 phim vào danh sách
      - Solution đầu tiên tìm được ở depth K
      - Không đảm bảo optimal (không dùng path cost)

    branch_limit: số actions xét tại mỗi node (giới hạn branching factor).
    Returns: (solution_state, stats)
    """
    t0 = time.time()

    if problem.goal_test(problem.initial):
        return problem.initial, _make_stats(0, t0, 0.0)

    # (state, g_cost)
    frontier = deque([(problem.initial, 0.0)])
    nodes_expanded = 0

    while frontier:
        node, g = frontier.popleft()
        nodes_expanded += 1

        for action in problem.actions(node)[:branch_limit]:
            child  = problem.result(node, action)
            new_g  = problem.path_cost(g, node, action, child)

            if problem.goal_test(child):
                return child, _make_stats(nodes_expanded, t0, new_g)

            # Chỉ mở rộng nếu chưa đến depth K-1
            if len(child) < problem.K:
                frontier.append((child, new_g))

        if nodes_expanded > 15_000:
            break

    return None, _make_stats(nodes_expanded, t0, None)


# ─────────────────────────────────────────────────────────────────────────────
# DFS
# ─────────────────────────────────────────────────────────────────────────────

def dfs(problem, branch_limit=15):
    """
    Depth-First Search: duyệt theo chiều sâu.

    Vì candidate_pool được sort by score giảm dần, DFS tương đương
    greedy search — tìm solution rất nhanh (K node expansions).
    Không đảm bảo optimal.

    Returns: (solution_state, stats)
    """
    t0 = time.time()

    stack = [(problem.initial, 0.0)]
    nodes_expanded = 0

    while stack:
        node, g = stack.pop()
        nodes_expanded += 1

        if problem.goal_test(node):
            return node, _make_stats(nodes_expanded, t0, g)

        actions = problem.actions(node)[:branch_limit]
        # Đẩy ngược để pop ra theo thứ tự thuận (high-score first)
        for action in reversed(actions):
            child = problem.result(node, action)
            new_g = problem.path_cost(g, node, action, child)
            stack.append((child, new_g))

        if nodes_expanded > 15_000:
            break

    return None, _make_stats(nodes_expanded, t0, None)


# ─────────────────────────────────────────────────────────────────────────────
# UCS
# ─────────────────────────────────────────────────────────────────────────────

def ucs(problem, branch_limit=20):
    """
    Uniform Cost Search: tìm solution với tổng chi phí nhỏ nhất.

    Với cost = −score, UCS tối đa hóa tổng relevance score → OPTIMAL.
    Khám phá nhiều node hơn DFS nhưng đảm bảo kết quả tốt nhất.

    Returns: (solution_state, stats)
    """
    t0 = time.time()

    counter = 0
    # (g_cost, tie_breaker, state)
    frontier = [(0.0, counter, problem.initial)]
    explored  = set()
    nodes_expanded = 0

    while frontier:
        g, _, node = heapq.heappop(frontier)

        if problem.goal_test(node):
            return node, _make_stats(nodes_expanded, t0, g)

        key = frozenset(node)
        if key in explored:
            continue
        explored.add(key)
        nodes_expanded += 1

        for action in problem.actions(node)[:branch_limit]:
            child = problem.result(node, action)
            new_g = problem.path_cost(g, node, action, child)
            counter += 1
            heapq.heappush(frontier, (new_g, counter, child))

        if nodes_expanded > 8_000:
            break

    return None, _make_stats(nodes_expanded, t0, None)
