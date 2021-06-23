__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pickle

import pytest

from jina import Document, DocumentArray, Flow

'''
User -> Train request -> RankTrainer Train -> RankTrainer Dump Weights/Parameters/Model ->
Ranker Load Model -> Re-rank
'''

cur_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def documents():
    """
    Imaging you're running an online shoppoing website, with features `brand`, `price`, `color`,
    label is `ndcg` (we want to optimize).
    """
    document_array = DocumentArray()
    document1 = Document(tags={'query_size': 35, 'query_price': 51, 'query_brand': 1})
    document1.matches.append(
        Document(
            tags={
                'match_size': 37.0,
                'match_price': 50,
                'match_brand': 1,
                'relevance': 3,
            }
        )
    )
    document1.matches.append(
        Document(
            tags={
                'match_size': 38.0,
                'match_price': 38,
                'match_brand': 1,
                'relevance': 5,
            }
        )
    )
    document1.matches.append(
        Document(
            tags={
                'match_size': 36.0,
                'match_price': 51,
                'match_brand': 2,
                'relevance': 2,
            }
        )
    )

    document2 = Document(tags={'query_size': 41, 'query_price': 80, 'query_brand': 3})
    document2.matches.append(
        Document(
            tags={
                'match_size': 42.0,
                'match_price': 106,
                'match_brand': 3,
                'relevance': 1,  # gain is low because expensive.
            }
        )
    )
    document2.matches.append(
        Document(
            tags={
                'match_size': 41.0,
                'match_price': 38.5,
                'match_brand': 3,
                'relevance': 7,
            }
        )
    )
    document2.matches.append(
        Document(
            tags={
                'match_size': 41.0,
                'match_price': 79.5,
                'match_brand': 2,
                'relevance': 6,
            }
        )
    )
    document_array.extend([document1, document2])
    return document_array


def test_train_offline(documents):
    """
    The objective of this test is to ensure ranker and trainer works as expected.
    Our data set consist of 2 features field, `price` and `size`. Label field is named as `relevance`.
    Before using ranker trainer, we manually train a linear model based on `price` field, use a
      Jina search flow to find documents and scores with the `doc_to_query`. Since the `price` of the `doc_to_query`
      has been set to 1, so the pre-trained model will always return the same value and all the scores will be the same.
      so we assert the length of prediction is 1 in `validate_ranking_by_price`.
    Afterwords, we fire a ranker trainer, it will dump a new model. The training set of the new model is based on `size`
      feature, see `docs_to_train`, and the `price` is not going to have any impact on the predictions. When we search
      the result with `doc_to_query`, we expect the relevance score keep increasing since the `size` in `doc_to_query`
      keeps increasing.
      see `validate_ranking_by_size`.
    """

    # def validate_ranking_by_price(req):
    #     pred = set()
    #     for match in req.docs[0].matches:
    #         pred.add(match.scores['num_clicks'])
    #     assert len(pred) == 1  # since price tag never changes, all scores are the same.
    #
    # def validate_ranking_by_size(req):
    #     pred = []
    #     for match in req.docs[0].matches:
    #         pred.append(match.scores['num_clicks'])
    #     print(f"\n{pred}\n")
    #     assert (
    #         sorted(pred, reverse=True) == pred
    #     )  # assure predictions are ordered since size increases

    def print_scores(resp):
        for doc in resp.docs:
            for match in doc.matches:
                print(f"\n\n{match.scores['relevance'].value}\n\n")

    with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
        f.post(on='/train', inputs=documents)

    with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
        f.post(on='/search', inputs=documents, on_done=print_scores)

    # Before Ranker Trainer, the feature is completely rely on `price` tag, `size` can be seen as a bias.

    # with Flow.load_config(os.path.join(cur_dir, 'flow_rank_train.yml')) as f:
    #     f.search(inputs=[documents], on_done=validate_ranking_by_price)

    # Run Ranker Trainer

    # # After Ranker Trainer, the feature should be completely rely on `size` tag.
    #
    # with Flow.load_config(os.path.join(cur_dir, 'flow_offline_search.yml')) as f:
    #     f.search(inputs=[doc_to_query], on_done=validate_ranking_by_size)
