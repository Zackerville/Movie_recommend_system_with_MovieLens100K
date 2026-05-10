"""[A] A* search với heuristic admissible."""

import time
import heapq


def astar(problem):
    """
    A* Search: tìm solution tối ưu bằng f(n) = g(n) + h(n).

    g(n) = path cost  = −(sum of scores đã chọn)
    h(n) = heuristic  = −(sum of top K−|n| remaining scores)
    f(n) = g + h      → minimize f ≡ maximize total relevance

    Tính chất:
      - Heuristic consistent → A* với closed list là OPTIMAL
      - Mở rộng ít node hơn UCS nhờ heuristic pruning
      - Độ phức tạp: O(b^d) worst case, nhưng thực tế rất nhanh

    Returns: (solution_state, stats)
        stats: {nodes_expanded, time_sec, path_cost}
    """
    t0      = time.time()
    counter = 0

    h0 = problem.heuristic(problem.initial)
    # (f_cost, tie_breaker, g_cost, state)
    frontier = [(h0, counter, 0.0, problem.initial)]
    explored  = set()
    nodes_expanded = 0

    while frontier:
        f, _, g, node = heapq.heappop(frontier)

        if problem.goal_test(node):
            return node, {
                'nodes_expanded': nodes_expanded,
                'time_sec':       round(time.time() - t0, 5),
                'path_cost':      round(g, 4),
            }

        key = frozenset(node)
        if key in explored:
            continue
        explored.add(key)
        nodes_expanded += 1

        for action in problem.actions(node):
            child = problem.result(node, action)
            new_g = problem.path_cost(g, node, action, child)
            h     = problem.heuristic(child)
            counter += 1
            heapq.heappush(frontier, (new_g + h, counter, new_g, child))

        if nodes_expanded > 8_000:
            break

    return None, {
        'nodes_expanded': nodes_expanded,
        'time_sec':       round(time.time() - t0, 5),
        'path_cost':      None,
    }
