"""[C] Rule engine với forward chaining cho hệ gợi ý phim."""


class Rule:
    """
    Một luật IF-THEN trong Knowledge Base.

    condition : callable(context: dict) -> bool
                True nếu luật này được kích hoạt
    action    : callable(context: dict, pool: list[dict]) -> list[dict]
                Biến đổi movie pool (filter, boost, rerank)
    """

    def __init__(self, name, condition, action, description=''):
        self.name        = name
        self.condition   = condition
        self.action      = action
        self.description = description

    def __repr__(self):
        return f'Rule({self.name!r})'


class KnowledgeBase:
    """
    Forward Chaining Engine.

    Áp dụng tất cả các luật thỏa điều kiện theo thứ tự vào movie pool.
    Mỗi luật có thể filter (loại bỏ), boost (tăng score), hoặc rerank pool.

    Context keys (chuẩn):
        user_id       : int
        age           : int
        gender        : 'M' | 'F'
        num_ratings   : int    — số phim đã đánh giá
        mean_rating   : float  — rating trung bình của user
        genre_pref    : dict {genre: float} — mean rating per genre
        genre_entropy : float  — đa dạng sở thích
        active_days   : float  — số ngày hoạt động
        watched_ids   : set    — movie_ids đã xem
        global_mean   : float  — global mean rating
        K             : int    — số phim cần gợi ý

    Pool item keys (chuẩn):
        movie_id      : int
        score         : float  — predicted relevance
        bayesian_avg  : float
        num_ratings   : int    — số người đánh giá
        year          : float
        genre_flags   : dict {genre: 0|1}
    """

    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        self.rules.append(rule)

    def add_rules(self, rules):
        self.rules.extend(rules)

    def forward_chain(self, context, movie_pool):
        """
        Áp dụng tất cả luật thỏa điều kiện, trả về (filtered_pool, fired_rules).

        Args:
            context    : dict thông tin user (xem KnowledgeBase docstring)
            movie_pool : list of movie dicts

        Returns:
            (movie_pool_after, fired_rule_names)
        """
        fired = []
        for rule in self.rules:
            try:
                if rule.condition(context):
                    movie_pool = rule.action(context, movie_pool)
                    fired.append(rule.name)
            except Exception:
                pass  # lỗi rule → bỏ qua, không làm crash hệ thống
        return movie_pool, fired

    def explain(self, context, movie_pool):
        """
        Giống forward_chain nhưng trả về trace chi tiết từng luật.

        Returns:
            (movie_pool_after, trace_list)
            trace_list: list of dicts với keys:
                rule, description, triggered, pool_before, pool_after
        """
        trace = []
        for rule in self.rules:
            try:
                triggered = rule.condition(context)
                n_before  = len(movie_pool)
                if triggered:
                    movie_pool = rule.action(context, movie_pool)
                trace.append({
                    'rule':        rule.name,
                    'description': rule.description,
                    'triggered':   triggered,
                    'pool_before': n_before,
                    'pool_after':  len(movie_pool),
                })
            except Exception as e:
                trace.append({
                    'rule':      rule.name,
                    'triggered': False,
                    'error':     str(e),
                })
        return movie_pool, trace
