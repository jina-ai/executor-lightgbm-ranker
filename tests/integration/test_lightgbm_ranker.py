__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os

import pytest
from jina import Flow

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def model_path():
    """
    Pretrained model to be stored in tmpdir.
    It will be trained on fake data forcing that the first feature correlates positively with relevance, while the rest are
    are random
    :param tmpdir:
    :return:
    """
    model_path = os.path.join('.', 'model.txt')
    return model_path


def test_train_offline(documents, model_path):
    # Step 1, without a model (train), we manually add score to the document
    # The ranking score is not affected by the model
    def validate_initial_ranking(resp):
        scores = []
        for doc in resp.docs:
            scores.append(doc.scores['relevance'].value)
        assert scores[0] < scores[1] < scores[2]

    for idx, document in enumerate(documents):
        document.scores['relevance'] = idx  # manually assign score to document
    with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
        f.post(on='/search', inputs=documents, on_done=validate_initial_ranking)

    # Step 2, train a model (train), will leave an model inside workdir
    assert not os.path.exists(model_path)
    with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
        f.post(on='/train', inputs=documents)
        f.post(on='/dump', parameters={'model_path': model_path})
    assert os.path.exists(model_path)  # after train, dump a model in model path

    # Step 3, call the search endpoint again, assure the relevance
    def validate_initial_ranking2(resp):
        scores = []
        for doc in resp.docs:
            for match in doc.matches:
                scores.append(match.scores['relevance'].value)
        assert len(scores) == 150
        assert scores[0] >= scores[1] >= scores[2] >= scores[-2] >= scores[-1]

    with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
        f.post(on='/search', inputs=documents, on_done=validate_initial_ranking2)
