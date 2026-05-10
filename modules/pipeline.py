"""[L.O.4] Pipeline tích hợp 5 thành phần: [E] DT + SVD + [D] NB → [C] KB → [A+B] A*."""

import os
import pickle
import numpy as np
import pandas as pd

from .utils.data_loader import load_all, GENRE_COLS
from .utils.features import build_features_for_inference
from .kb.rules import build_kb
from .search.problem import RecommendationProblem
from .search.astar import astar


class RecommenderPipeline:
    """Pipeline gợi ý phim: blended = 0.55·DT + 0.25·SVD + 0.20·NB → KB → A*."""

    COLD_START_THRESHOLD = 5
    W_DT  = 0.55
    W_SVD = 0.25
    W_NB  = 0.20

    def __init__(self, data_dir, features_dir,
                 nb_model_path=None, ml_model_path=None):
        self.data_dir     = data_dir
        self.features_dir = features_dir

        self.ratings, self.movies, self.users = load_all(data_dir)
        self.global_mean = float(self.ratings['rating'].mean())

        self.user_latent  = np.load(os.path.join(features_dir, 'user_latent.npy'))
        self.movie_latent = np.load(os.path.join(features_dir, 'movie_latent.npy'))
        self.train_df     = pd.read_csv(os.path.join(features_dir, 'train_ratings.csv'))
        self.test_df      = pd.read_csv(os.path.join(features_dir, 'test_ratings.csv'))

        self._user_means = (
            self.train_df.groupby('user_id')['rating'].mean()
            .reindex(range(1, 944), fill_value=self.global_mean)
            .values
        )

        movie_stats = (
            self.train_df.groupby('movie_id')['rating']
            .agg(avg_rating='mean', num_ratings_m='count')
            .reset_index()
        )
        C = 50
        movie_stats['bayesian_avg'] = (
            (movie_stats['num_ratings_m'] * movie_stats['avg_rating']
             + C * self.global_mean)
            / (movie_stats['num_ratings_m'] + C)
        )
        movies_ext = self.movies.merge(movie_stats, on='movie_id', how='left').fillna(
            {'avg_rating': self.global_mean, 'num_ratings_m': 0,
             'bayesian_avg': self.global_mean}
        )
        self._movie_info = movies_ext.set_index('movie_id').to_dict('index')

        u_stats = (
            self.train_df.groupby('user_id')['rating']
            .agg(mean_rating='mean', num_ratings='count')
            .reset_index()
        )
        u_full = self.users.merge(u_stats, on='user_id', how='left').fillna(
            {'mean_rating': self.global_mean, 'num_ratings': 0}
        )

        rg = self.train_df.merge(
            self.movies[['movie_id'] + GENRE_COLS], on='movie_id', how='left'
        )
        rg_long = rg.melt(id_vars=['user_id', 'rating'],
                          value_vars=GENRE_COLS, var_name='genre', value_name='has_genre')
        rg_long = rg_long[rg_long['has_genre'] == 1]
        genre_pref_df = (
            rg_long.pivot_table(index='user_id', columns='genre',
                                values='rating', aggfunc='mean')
            .reindex(columns=GENRE_COLS).fillna(self.global_mean)
        )

        ts = (self.train_df.groupby('user_id')['timestamp']
              .agg(first_ts='min', last_ts='max').reset_index())
        ts['active_days'] = ((ts['last_ts'] - ts['first_ts']) / 86400).clip(lower=0)

        self._user_info     = u_full.set_index('user_id').to_dict('index')
        self._genre_pref_df = genre_pref_df
        self._ts_info       = ts.set_index('user_id').to_dict('index')

        self._watched = (
            self.train_df.groupby('user_id')['movie_id']
            .apply(set).to_dict()
        )

        self.kb = build_kb(global_mean=self.global_mean)

        self.nb_model = None
        self.ml_model = None
        if nb_model_path and os.path.exists(nb_model_path):
            with open(nb_model_path, 'rb') as f:
                self.nb_model = pickle.load(f)
        if ml_model_path and os.path.exists(ml_model_path):
            with open(ml_model_path, 'rb') as f:
                self.ml_model = pickle.load(f)

        print(f"Pipeline ready  |  global_mean={self.global_mean:.3f}  "
              f"DT={'yes' if self.ml_model else 'no'}  "
              f"NB={'yes' if self.nb_model else 'no'}")

    def _svd_scores_all(self, user_id):
        """SVD predicted rating cho toàn bộ 1682 phim."""
        uid_idx = user_id - 1
        return self.movie_latent @ self.user_latent[uid_idx] + self._user_means[uid_idx]

    def _dt_scores(self, user_id, movie_ids):
        """[E] Custom DT predicted rating (1–5)."""
        X = build_features_for_inference(
            user_id, movie_ids,
            self.train_df, self.users, self.movies,
            global_mean=self.global_mean,
            precomputed_movie_stats=self._movie_info,
        )
        return np.clip(self.ml_model.predict(X), 1.0, 5.0)

    def _nb_proba(self, user_id, movie_ids):
        """[D] NaiveBayes P(like=1)."""
        if self.nb_model is None:
            return np.full(len(movie_ids), 0.5)
        from .bayes.naive_bayes import build_nb_features
        fake = pd.DataFrame({
            'user_id':  [user_id] * len(movie_ids),
            'movie_id': list(movie_ids),
            'rating':   [self.global_mean] * len(movie_ids),
        })
        try:
            X, _ = build_nb_features(fake, self.users, self.movies,
                                      global_mean=self.global_mean)
            proba   = self.nb_model.predict_proba(X)
            pos_idx = int(np.where(self.nb_model.classes_ == 1)[0][0])
            return proba[:, pos_idx]
        except Exception:
            return np.full(len(movie_ids), 0.5)

    @staticmethod
    def _minmax(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-9)

    def _score_candidates(self, user_id, top_n=150):
        """[E+D] Blend DT + SVD + NB → top_n candidates."""
        watched  = self._watched.get(user_id, set())
        all_ids  = np.arange(1, 1683)
        mask     = np.array([m not in watched for m in all_ids])
        cand_ids = all_ids[mask]

        svd_all  = self._svd_scores_all(user_id)
        svd_cand = svd_all[mask]

        if self.ml_model is not None:
            dt_cand = self._dt_scores(user_id, cand_ids)
        else:
            dt_cand = svd_cand.copy()

        nb_cand  = self._nb_proba(user_id, cand_ids)
        dt_norm  = self._minmax(dt_cand)
        svd_norm = self._minmax(svd_cand)
        blended  = self.W_DT * dt_norm + self.W_SVD * svd_norm + self.W_NB * nb_cand

        top_idx = np.argsort(blended)[::-1][:top_n]
        pool = []
        for i in top_idx:
            mid  = int(cand_ids[i])
            info = self._movie_info.get(mid, {})
            pool.append({
                'movie_id':    mid,
                'score':       float(blended[i]),
                'bayesian_avg': float(info.get('bayesian_avg', self.global_mean)),
                'num_ratings': int(info.get('num_ratings_m', 0)),
                'year':        float(info.get('year', 1990)),
                'title':       str(info.get('title', f'Movie {mid}')),
                'genre_flags': {g: int(info.get(g, 0)) for g in GENRE_COLS},
            })
        return pool

    def _popularity_scores(self, user_id, top_n=150):
        """Cold-start fallback: score = bayesian_avg + 0.3·log(popularity)."""
        watched = self._watched.get(user_id, set())
        pool = []
        for mid, info in self._movie_info.items():
            if mid in watched:
                continue
            score = (float(info.get('bayesian_avg', self.global_mean))
                     + 0.3 * np.log1p(info.get('num_ratings_m', 0)))
            pool.append({
                'movie_id':    int(mid),
                'score':       score,
                'bayesian_avg': float(info.get('bayesian_avg', self.global_mean)),
                'num_ratings': int(info.get('num_ratings_m', 0)),
                'year':        float(info.get('year', 1990)),
                'title':       str(info.get('title', f'Movie {mid}')),
                'genre_flags': {g: int(info.get(g, 0)) for g in GENRE_COLS},
            })
        pool.sort(key=lambda x: x['score'], reverse=True)
        return pool[:top_n]

    def _build_context(self, user_id):
        ui = self._user_info.get(user_id, {})
        gp = (self._genre_pref_df.loc[user_id].to_dict()
              if user_id in self._genre_pref_df.index else {})
        ts = self._ts_info.get(user_id, {})

        gp_vals = np.array(list(gp.values())) if gp else np.array([self.global_mean])
        gp_norm = gp_vals - gp_vals.min() + 1e-9
        gp_norm /= gp_norm.sum()
        genre_entropy = float(-np.sum(gp_norm * np.log(gp_norm + 1e-9)))

        return {
            'user_id':       user_id,
            'age':           int(ui.get('age', 30)),
            'gender':        str(ui.get('gender', 'M')),
            'num_ratings':   int(ui.get('num_ratings', 0)),
            'mean_rating':   float(ui.get('mean_rating', self.global_mean)),
            'genre_pref':    gp,
            'genre_entropy': genre_entropy,
            'active_days':   float(ts.get('active_days', 0)),
            'watched_ids':   self._watched.get(user_id, set()),
            'global_mean':   self.global_mean,
            'K':             10,
        }

    def recommend(self, user_id: int, K: int = 10, verbose: bool = False):
        """Gợi ý Top-K phim cho user_id. Trả về list[dict] với rank, title, score, genres."""
        is_cold = (self._user_info.get(user_id, {})
                   .get('num_ratings', 0) < self.COLD_START_THRESHOLD)

        if is_cold:
            pool = self._popularity_scores(user_id, top_n=150)
            if verbose:
                print(f"[Cold-start] user={user_id} → popularity scoring")
        else:
            pool = self._score_candidates(user_id, top_n=150)
            if verbose:
                print(f"[DT+SVD+NB] user={user_id} → pool={len(pool)} candidates")

        context = self._build_context(user_id)
        context['K'] = K
        pool, fired_rules = self.kb.forward_chain(context, pool)
        if verbose:
            print(f"[KB] fired={fired_rules}  pool_after={len(pool)}")

        if len(pool) < K:
            fallback = self._popularity_scores(user_id, top_n=50)
            existing = {m['movie_id'] for m in pool}
            for m in fallback:
                if m['movie_id'] not in existing:
                    pool.append(m)
                if len(pool) >= K * 2:
                    break

        pool_sorted = sorted(pool, key=lambda x: x['score'], reverse=True)
        astar_pool  = [m['movie_id'] for m in pool_sorted[:min(len(pool), 50)]]
        scores_dict = {m['movie_id']: m['score'] for m in pool}

        problem         = RecommendationProblem(astar_pool, scores_dict, K=K)
        solution, stats = astar(problem)
        if verbose:
            print(f"[A*] nodes={stats['nodes_expanded']}  time={stats['time_sec']:.3f}s")

        if solution is None or len(solution) < K:
            solution = tuple(m['movie_id'] for m in pool_sorted[:K])

        movie_dict = {m['movie_id']: m for m in pool}
        results = []
        for rank, mid in enumerate(solution, 1):
            info   = movie_dict.get(mid, {})
            mi     = self._movie_info.get(mid, {})
            genres = [g for g in GENRE_COLS if info.get('genre_flags', {}).get(g, 0)]
            results.append({
                'rank':          rank,
                'movie_id':      mid,
                'title':         info.get('title', mi.get('title', f'Movie {mid}')),
                'score':         round(float(info.get('score', 0)), 4),
                'genres':        genres,
                'year':          int(info.get('year', mi.get('year', 0)) or 0),
                'bayesian_avg':  round(float(info.get('bayesian_avg', self.global_mean)), 3),
                'rules_applied': fired_rules,
                'cold_start':    is_cold,
            })
        return results

    def explain(self, user_id: int, K: int = 10):
        """KB explanation trace cho user_id."""
        is_cold = (self._user_info.get(user_id, {})
                   .get('num_ratings', 0) < self.COLD_START_THRESHOLD)
        pool    = (self._popularity_scores(user_id) if is_cold
                   else self._score_candidates(user_id))
        context = self._build_context(user_id)
        context['K'] = K
        _, trace = self.kb.explain(context, pool)
        return trace

    def svd_rmse_on_test(self):
        """SVD RMSE trên test set."""
        preds, trues = [], []
        for _, row in self.test_df.iterrows():
            uid, mid = int(row['user_id']), int(row['movie_id'])
            pred = float(self.user_latent[uid - 1] @ self.movie_latent[mid - 1]
                         + self._user_means[uid - 1])
            preds.append(np.clip(pred, 1.0, 5.0))
            trues.append(float(row['rating']))
        return round(float(np.sqrt(np.mean((np.array(trues) - np.array(preds)) ** 2))), 4)
