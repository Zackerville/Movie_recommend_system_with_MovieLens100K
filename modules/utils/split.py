"""Train/test split 80/20 stratified per user."""

import pandas as pd
import numpy as np


def train_test_split(ratings, test_ratio=0.2, random_state=42):
    """
    Chia train/test 80/20 theo từng user (stratified per-user).

    Mỗi user đóng góp đúng test_ratio phần trăm rating vào test set.
    Đảm bảo mỗi user có ít nhất 1 rating trong test (kể cả user ít rating).

    Args:
        ratings (DataFrame): toàn bộ ratings.
        test_ratio (float):  tỉ lệ test, mặc định 0.2.
        random_state (int):  seed cho reproducibility.

    Returns:
        train_df, test_df
    """
    rng = np.random.RandomState(random_state)
    train_list, test_list = [], []

    for user_id, group in ratings.groupby('user_id'):
        group = group.sample(frac=1, random_state=int(rng.randint(0, 9999)))
        n_test = max(1, int(len(group) * test_ratio))
        test_list.append(group.iloc[:n_test])
        train_list.append(group.iloc[n_test:])

    train = pd.concat(train_list).reset_index(drop=True)
    test  = pd.concat(test_list).reset_index(drop=True)

    print(f"Train: {len(train):,} ratings ({len(train)/len(ratings):.1%})")
    print(f"Test : {len(test):,} ratings ({len(test)/len(ratings):.1%})")
    print(f"Users in train: {train['user_id'].nunique()} | test: {test['user_id'].nunique()}")
    return train, test
