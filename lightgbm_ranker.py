__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict, Optional, Tuple, List, Union

import lightgbm
import numpy as np
from jina import Executor, DocumentArray, requests
from jina.excepts import PretrainedModelFileDoesNotExist


class LightGBMRanker(Executor):
    """
    Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)
    :param model_path: path to the pretrained model previously trained using LightGBM
    :param query_features: name of the features to extract from query Documents and used to compute relevance scores by the model loaded
    from model_path
    :param match_features: name of the features to extract from match Documents and used to compute relevance scores by the model loaded
    from model_path
    :param query_categorical_features: name of features contained in `query_features` corresponding to categorical features.
    :param match_categorical_features: name of features contained in `match_features` corresponding to categorical features.
    :param query_features_before: True if `query_features` must be placed before the `match` ones in the `dataset` used for prediction.
    :param args: Additional positional arguments
    :param kwargs: Additional keyword arguments
    .. note::
        The name of the features are used to extract the features from incoming `documents`. Check how these features are accessed in
        :class:`Document` at https://docs.jina.ai/api/jina.types.document/
    """

    def __init__(
        self,
        model_path: Optional[str] = 'tmp/model.txt',
        params: Dict = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'lambdarank',
        },
        query_features: Tuple[str] = [
            'query_length',
            'query_language',
        ],
        match_features: Tuple[str] = [
            'document_length',
            'document_language',
            'document_pagerank',
        ],
        label_feature: str = 'relevance',
        categorical_query_features: Optional[List[str]] = None,
        categorical_match_features: Optional[List[str]] = None,
        query_features_before: bool = True,
        *args,
        **kwargs,
    ):
        super(LightGBMRanker, self).__init__(*args, **kwargs)
        self.params = params
        self.model_path = model_path
        self.query_features = query_features
        self.match_features = match_features
        self.categorical_query_features = categorical_query_features
        self.categorical_match_features = categorical_match_features
        self.label_feature = label_feature
        self.query_features_before = query_features_before
        if self.model_path and os.path.exists(self.model_path):
            self.booster = lightgbm.Booster(model_file=self.model_path)
            model_num_features = self.booster.num_feature()
            expected_num_features = len(self.query_features + self.match_features)
            if model_num_features != expected_num_features:
                raise ValueError(
                    f'The number of features expected by the LightGBM model {model_num_features} is different'
                    f'than the ones provided in input {expected_num_features}'
                )
        else:
            raise PretrainedModelFileDoesNotExist(
                f'model {self.model_path} does not exist'
            )

    def _get_features_dataset(
        self,
        docs: DocumentArray,
    ) -> 'lightgbm.Dataset':
        q_features, m_features, group = [], [], []
        for doc in docs:
            query_feature = []
            match_feature = []
            query_values = [doc.tags.get(feature) for feature in self.query_features]
            for match in doc.matches:
                match_values = [
                    match.tags.get(feature) for feature in self.match_features
                ]
                match_feature.append(match_values)
                query_feature.append(query_values)
            group.append(len(doc.matches))
            q_features.append(query_feature)
            m_features.append(match_feature)

        query_features = np.vstack(q_features)
        query_dataset = lightgbm.Dataset(
            data=query_features,
            group=group,
            feature_name=self.query_features,
            categorical_feature=self.categorical_query_features,
            free_raw_data=False,
        )

        match_features = np.vstack(m_features)
        match_dataset = lightgbm.Dataset(
            data=match_features,
            group=group,
            feature_name=self.match_features,
            categorical_feature=self.categorical_match_features,
            free_raw_data=False,
        )
        if self.query_features_before:
            return query_dataset.construct().add_features_from(
                match_dataset.construct()
            )
        else:
            return match_dataset.construct().add_features_from(
                query_dataset.construct()
            )

    @requests(on='/train')
    def train(self, docs: DocumentArray, **kwargs):
        """The :meth:`train` endpoint allows user to train ther lightgbm ranker
        in an incremental manner. The features will be extracted from the `attr`:`tags`,
        including all the
        """
        train_set = self._get_features_dataset(docs)
        self.booster = lightgbm.train(
            train_set=train_set,
            init_model=self.booster,
            params=self.params,
            keep_training_booster=True,
        )
        self.booster.save_model(self.model_path)

    @requests(on='/search')
    def rank(self, docs: DocumentArray, **kwargs):
        dataset = self._get_features_dataset(docs)
        predictions = self.booster.predict(dataset.get_data())
        for prediction, doc in zip(predictions, docs):
            doc.scores[self.label_feature] = prediction
