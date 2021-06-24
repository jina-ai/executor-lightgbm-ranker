__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest

from jinahub.ranker.lightgbm_ranker import LightGBMRanker


cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def model_path(tmpdir):
    """
    Pretrained model to be stored in tmpdir.
    It will be trained on fake data forcing that the first feature correlates positively with relevance, while the rest are
    are random
    :param tmpdir:
    :return:
    """
    model_path = os.path.join('.', 'model.txt')
    return model_path


@pytest.fixture
def param():
    return {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'metric_freq': 1,
        'is_training_metric': True,
        'max_bin': 255,
        'num_trees': 20,
        'learning_rate': 0.01,
        'num_leaves': 31,
        'tree_learner': 'serial',
        'feature_fraction': 1.0,
        'min_data_in_leaf': 1,
        'min_sum_hessian_in_leaf': 5.0,
        'is_enable_sparse': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'force_col_wise': True,
    }


# def test_lightgbm_rank(model_path, documents):
#     ranker = LightGBMRanker(
#         model_path=model_path,
#         query_features=['query_price', 'query_size'],
#         match_features=[
#             'match_price',
#             'match_size',
#             'match_brand',
#         ],
#     )
#     ranker.rank(docs=documents)
#     for doc in documents:
#         for match in doc.matches:
#             assert match.scores['relevance']
#             print(match.scores)


def test_lightgbm_train(model_path, documents, param):
    ranker = LightGBMRanker(
        model_path=model_path,
        query_features=['query_price', 'query_size'],
        match_features=[
            'match_price',
            'match_size',
            'match_brand',
        ],
        params=param,
    )
    ranker.train(docs=documents)
    for doc in documents:
        for match in doc.matches:
            print(match.scores)
