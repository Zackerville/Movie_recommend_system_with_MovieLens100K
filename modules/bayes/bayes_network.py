"""[D] (Optional) Bayesian Network bằng pgmpy."""

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


def build_movie_preference_bn():
    """
    Xây dựng Bayesian Network mô hình hóa P(Like | Genre, Popularity, UserType).

    Cấu trúc:
        Genre      → Like
        Popularity → Like
        UserType   → Like

    Nodes:
        Genre      : {Drama, Action, Comedy, Other}
        Popularity : {Low, High}          (num_ratings ≤ 50 vs > 50)
        UserType   : {Harsh, Normal, Generous}
        Like       : {0, 1}

    CPD được ước lượng từ kinh nghiệm (domain knowledge) về MovieLens.
    """
    if not PGMPY_AVAILABLE:
        print('pgmpy chưa cài. Chạy: pip install pgmpy')
        return None

    model = BayesianNetwork([
        ('Genre',      'Like'),
        ('Popularity', 'Like'),
        ('UserType',   'Like'),
    ])

    # ── CPD: Genre (prior) ────────────────────────────────────────────────────
    cpd_genre = TabularCPD(
        variable='Genre', variable_card=4,
        values=[[0.33], [0.27], [0.23], [0.17]],
        state_names={'Genre': ['Drama', 'Action', 'Comedy', 'Other']},
    )

    # ── CPD: Popularity (prior) ───────────────────────────────────────────────
    cpd_pop = TabularCPD(
        variable='Popularity', variable_card=2,
        values=[[0.45], [0.55]],
        state_names={'Popularity': ['Low', 'High']},
    )

    # ── CPD: UserType (prior) ─────────────────────────────────────────────────
    cpd_user = TabularCPD(
        variable='UserType', variable_card=3,
        values=[[0.22], [0.51], [0.27]],
        state_names={'UserType': ['Harsh', 'Normal', 'Generous']},
    )

    # ── CPD: Like | Genre, Popularity, UserType ───────────────────────────────
    # 4 × 2 × 3 = 24 parent combinations
    # P(Like=1) = clip(genre_base + pop_bonus + user_bonus, 0.05, 0.95)
    genre_base = {'Drama': 0.42, 'Action': 0.38, 'Comedy': 0.44, 'Other': 0.35}
    pop_bonus  = {'Low': 0.00, 'High': 0.10}
    user_bonus = {'Harsh': -0.16, 'Normal': 0.00, 'Generous': +0.16}

    genres  = ['Drama', 'Action', 'Comedy', 'Other']
    pops    = ['Low', 'High']
    utypes  = ['Harsh', 'Normal', 'Generous']

    p1_list = []
    for g in genres:
        for p in pops:
            for u in utypes:
                plike = genre_base[g] + pop_bonus[p] + user_bonus[u]
                p1_list.append(float(max(0.05, min(0.95, plike))))

    p0_list = [1.0 - p for p in p1_list]

    cpd_like = TabularCPD(
        variable='Like', variable_card=2,
        values=[p0_list, p1_list],
        evidence=['Genre', 'Popularity', 'UserType'],
        evidence_card=[4, 2, 3],
        state_names={
            'Like':       [0, 1],
            'Genre':      genres,
            'Popularity': pops,
            'UserType':   utypes,
        },
    )

    model.add_cpds(cpd_genre, cpd_pop, cpd_user, cpd_like)
    assert model.check_model(), 'Bayesian Network không hợp lệ!'

    return model


def query_bn(model, evidence: dict, query_var: str = 'Like'):
    """
    Query P(query_var | evidence) bằng Variable Elimination.

    Ví dụ:
        evidence = {'Genre': 'Drama', 'Popularity': 'High', 'UserType': 'Generous'}
        → P(Like=1 | evidence)

    Returns:
        pgmpy DiscreteFactor hoặc None nếu model là None
    """
    if model is None:
        return None
    infer  = VariableElimination(model)
    result = infer.query([query_var], evidence=evidence, show_progress=False)
    return result


def demo_queries(model):
    """Minh họa 3 truy vấn xác suất điển hình."""
    if model is None:
        print('Model chưa được build.')
        return

    queries = [
        {'Genre': 'Drama',  'Popularity': 'High', 'UserType': 'Generous'},
        {'Genre': 'Action', 'Popularity': 'Low',  'UserType': 'Harsh'},
        {'Genre': 'Comedy', 'Popularity': 'High', 'UserType': 'Normal'},
    ]

    for i, ev in enumerate(queries, 1):
        result = query_bn(model, ev)
        if result is not None:
            p_like = result.values[1]
            print(f'Query {i}: P(Like=1 | {ev}) = {p_like:.4f}')
