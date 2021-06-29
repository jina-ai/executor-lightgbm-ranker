import pytest

from jina import Document, DocumentArray


@pytest.fixture
def documents():
    """
    Imaging you're running an online shoppoing website, with features `brand`, `price`, `color`,
    label is `relevance` (we want to optimize).
    """
    document_array = DocumentArray()
    document1 = Document(tags={'query_size': 35, 'query_price': 31, 'query_brand': 1})
    for i in range(0, 50):
        # assume increase the price will decrease the relevance
        d = Document(
            tags={
                'match_size': 35,
                'match_price': 31 + 2 * i,
                'match_brand': 1,
                'relevance': int((100 - i) / 10),
            }
        )
        document1.matches.append(d)

    document2 = Document(tags={'query_size': 41, 'query_price': 80, 'query_brand': 3})
    for i in range(0, 50):
        # assume increase the price will decrease the relevance
        d = Document(
            tags={
                'match_size': 41,
                'match_price': 80 + 2 * i,
                'match_brand': 1,
                'relevance': int((80 - i) / 10),
            }
        )
        document2.matches.append(d)

    document3 = Document(tags={'query_size': 39, 'query_price': 60, 'query_brand': 3})
    for i in range(0, 50):
        # assume increase the price will decrease the relevance
        d = Document(
            tags={
                'match_size': 39,
                'match_price': 60 + 2 * i,
                'match_brand': 1,
                'relevance': int((90 - i) / 10),
            }
        )
        document3.matches.append(d)

    document_array.extend([document1, document2, document3])
    return document_array
