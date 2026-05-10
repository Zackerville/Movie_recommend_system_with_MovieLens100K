"""[C] 20 luật IF-THEN cho hệ gợi ý phim (Knowledge Base)."""

from .engine import Rule, KnowledgeBase

def _boost(pool, genre, factor=1.25):
    """Tăng score của phim có thể loại genre."""
    result = []
    for m in pool:
        if m.get('genre_flags', {}).get(genre, 0):
            m = dict(m)
            m['score'] = m['score'] * factor
        result.append(m)
    return result

def _boost_multi(pool, genres, factor=1.2):
    """Tăng score nếu phim có ít nhất 1 trong các genre."""
    result = []
    for m in pool:
        flags = m.get('genre_flags', {})
        if any(flags.get(g, 0) for g in genres):
            m = dict(m)
            m['score'] = m['score'] * factor
        result.append(m)
    return result

def _filter_quality(pool, min_bayesian, min_pool=10):
    """Filter phim có bayesian_avg < min_bayesian. Giữ ít nhất min_pool."""
    filtered = [m for m in pool if m.get('bayesian_avg', 0) >= min_bayesian]
    return filtered if len(filtered) >= min_pool else pool

def _pref(ctx, genre):
    """Trả về genre preference score của user, mặc định = global_mean."""
    return ctx.get('genre_pref', {}).get(genre, ctx.get('global_mean', 3.53))

def build_kb(global_mean: float = 3.53) -> KnowledgeBase:
    """
    Xây dựng KnowledgeBase với 20 luật IF-THEN.

    Các nhóm luật:
        [1]   Anti-watched        — luôn loại phim đã xem
        [2-3] User activity       — cold-start, harsh rater
        [4-11] Genre preferences  — Drama, Comedy, Action, Sci-Fi, Romance,
                                    Horror, Adventure, Thriller
        [12-13] Age-based         — young (<25), senior (>55)
        [14]   Gender preference  — phim phù hợp giới tính
        [15-16] Taste profile     — diverse taste, niche seeker
        [17]   Recent movies      — user hoạt động lâu thích phim mới
        [18]   Classic movies     — user thích phim kinh điển
        [19]   Quality boost      — phim chất lượng cao
        [20]   Safety guard       — không để pool rỗng
    """
    gm = global_mean
    kb = KnowledgeBase()

    kb.add_rule(Rule(
        name='anti_watched',
        condition=lambda ctx: True,
        action=lambda ctx, pool: [
            m for m in pool if m['movie_id'] not in ctx.get('watched_ids', set())
        ],
        description='Loai bo phim user da xem'
    ))

    kb.add_rule(Rule(
        name='cold_start_popular',
        condition=lambda ctx: ctx.get('num_ratings', 0) < 5,
        action=lambda ctx, pool: sorted(
            pool,
            key=lambda m: m['score'] + 0.3 * min(m.get('num_ratings', 0) / 500.0, 1.0),
            reverse=True
        ),
        description='Cold-start: uu tien phim pho bien cho user moi'
    ))

    kb.add_rule(Rule(
        name='harsh_rater_quality',
        condition=lambda ctx, _gm=gm: ctx.get('mean_rating', _gm) < _gm - 0.4,
        action=lambda ctx, pool, _gm=gm: _filter_quality(pool, _gm + 0.25),
        description='User kho tinh: chi goi y phim co bayesian_avg > global_mean+0.25'
    ))

    genre_rules = [
        ('Drama',     0.40, 1.28),
        ('Comedy',    0.40, 1.25),
        ('Action',    0.35, 1.25),
        ('Sci-Fi',    0.30, 1.22),
        ('Romance',   0.40, 1.25),
        ('Horror',    0.25, 1.20),
        ('Adventure', 0.35, 1.22),
        ('Thriller',  0.30, 1.22),
    ]
    for genre, threshold, factor in genre_rules:
        _g, _t, _f = genre, threshold, factor
        kb.add_rule(Rule(
            name=f'{_g.lower().replace("-", "_")}_fan',
            condition=lambda ctx, g=_g, t=_t, _gm=gm: _pref(ctx, g) > _gm + t,
            action=lambda ctx, pool, g=_g, f=_f: _boost(pool, g, f),
            description=f'Fan {_g}: tang diem phim {_g} x{_f}'
        ))

    kb.add_rule(Rule(
        name='young_user_pref',
        condition=lambda ctx: ctx.get('age', 30) < 25,
        action=lambda ctx, pool: _boost_multi(
            pool, ["Animation", "Children's", "Comedy"], factor=1.20
        ),
        description='User tre (<25): boost Animation, Childrens, Comedy'
    ))

    kb.add_rule(Rule(
        name='senior_user_pref',
        condition=lambda ctx: ctx.get('age', 30) > 55,
        action=lambda ctx, pool: _boost_multi(
            pool, ['Documentary', 'Drama', 'War'], factor=1.20
        ),
        description='User lon tuoi (>55): boost Documentary, Drama, War'
    ))

    kb.add_rule(Rule(
        name='gender_pref',
        condition=lambda ctx: ctx.get('gender') in ('M', 'F'),
        action=lambda ctx, pool: _boost_multi(
            pool,
            ['Action', 'Sci-Fi', 'Thriller'] if ctx.get('gender') == 'M'
            else ['Romance', 'Drama', 'Musical'],
            factor=1.10
        ),
        description='Gioi tinh: boost Action/SciFi/Thriller (Nam) hoac Romance/Drama (Nu)'
    ))

    def _diverse_boost(ctx, pool):
        from collections import Counter
        genre_count = Counter()
        for m in pool[:30]:
            for g, v in m.get('genre_flags', {}).items():
                if v:
                    genre_count[g] += 1
        rare = {g for g, c in genre_count.items() if c <= 3}
        result = []
        for m in pool:
            if any(m.get('genre_flags', {}).get(g, 0) for g in rare):
                m = dict(m)
                m['score'] *= 1.15
            result.append(m)
        return result

    kb.add_rule(Rule(
        name='diverse_taste',
        condition=lambda ctx: ctx.get('genre_entropy', 0) > 2.5,
        action=_diverse_boost,
        description='So thich da dang (entropy>2.5): tang diem the loai it xuat hien'
    ))

    kb.add_rule(Rule(
        name='niche_seeker',
        condition=lambda ctx: ctx.get('genre_entropy', 0) > 3.0,
        action=lambda ctx, pool: [
            dict(m, score=m['score'] * 1.18)
            if m.get('num_ratings', 100) < 30 else m
            for m in pool
        ],
        description='Niche seeker (entropy>3.0): boost phim it pho bien (<30 ratings)'
    ))

    kb.add_rule(Rule(
        name='recent_movies',
        condition=lambda ctx: ctx.get('active_days', 0) > 300,
        action=lambda ctx, pool: [
            dict(m, score=m['score'] * 1.12)
            if m.get('year', 1990) >= 1993 else m
            for m in pool
        ],
        description='User hoat dong lau: boost phim moi (>= 1993)'
    ))

    kb.add_rule(Rule(
        name='classic_fan',
        condition=lambda ctx, _gm=gm: (
            ctx.get('num_ratings', 0) > 50 and
            ctx.get('mean_rating', _gm) > _gm + 0.2
        ),
        action=lambda ctx, pool: [
            dict(m, score=m['score'] * 1.15)
            if m.get('year', 1990) < 1975 else m
            for m in pool
        ],
        description='User active + rating cao: boost phim kinh dien (< 1975)'
    ))

    # ── Luật 19: Quality boost — boost phim có bayesian_avg > global + 0.5 ───
    kb.add_rule(Rule(
        name='quality_boost',
        condition=lambda ctx, _gm=gm: ctx.get('mean_rating', _gm) > _gm + 0.3,
        action=lambda ctx, pool, _gm=gm: [
            dict(m, score=m['score'] * 1.10)
            if m.get('bayesian_avg', _gm) > _gm + 0.5 else m
            for m in pool
        ],
        description='User generous: boost phim chat luong cao (bayesian > global+0.5)'
    ))

    kb.add_rule(Rule(
        name='min_pool_guard',
        condition=lambda ctx: True,
        action=lambda ctx, pool: pool if pool else [],
        description='An toan: giu nguyen pool neu bi rong'
    ))

    return kb
