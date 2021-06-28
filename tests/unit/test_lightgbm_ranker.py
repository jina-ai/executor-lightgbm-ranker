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
    model_path = os.path.join(tmpdir, 'model.txt')
    return model_path


@pytest.fixture
def param():
    return {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'min_data_in_leaf': 1,
        'feature_pre_filter': False,
    }


def test_lightgbm_train(model_path, documents, param):
    """First assert model not exist, then train the model, and use the model to rank."""
    assert not os.path.exists(model_path)
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
    ranker.dump(parameters={'model_path': model_path})
    assert os.path.exists(model_path)  # assert model was trained

    ranker = LightGBMRanker(
        model_path=model_path,
        query_features=['query_price', 'query_size'],
        match_features=[
            'match_price',
            'match_size',
            'match_brand',
        ],
    )
    scores = []
    ranker.rank(docs=documents)
    for doc in documents:
        for match in doc.matches:
            assert match.scores['relevance']
            scores.append(match.scores['relevance'].value)
    assert len(scores) == 150
    assert len(set(scores)) == 3  # we have 3 queries.
