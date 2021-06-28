__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
from typing import Dict, Optional, Tuple, List

import lightgbm
import numpy as np
from jina.logging.logger import JinaLogger
from jina import Executor, DocumentArray, requests


logger = JinaLogger(context='LightGBMRanker')


class LightGBMRanker(Executor):
    """
    Computes a relevance score for each match using a pretrained Ltr model trained with LightGBM (https://lightgbm.readthedocs.io/en/latest/index.html)
    :param model_path: path to the pretrained model previously trained using LightGBM.
    :param params: Parameters used to train the LightGBM learning-to-rank model.
    :param query_features: name of the features to extract from query Documents and used to compute relevance scores by the model loaded
    from model_path
    :param match_features: name of the features to extract from match Documents and used to compute relevance scores by the model loaded
    from model_path
    :param label: If call :meth:`train` endpoint, the label will be used as groundtruth for model training. If on :meth:`rank` endpoint, the
    label will be used to assign a score to :attr:`Document.scores` field.
    :param categorical_query_features: name of features contained in `query_features` corresponding to categorical features.
    :param categorical_match_features: name of features contained in `match_features` corresponding to categorical features.
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
            'min_data_in_leaf': 1,
            'feature_pre_filter': False,
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
        label: str = 'relevance',
        categorical_query_features: Optional[List[str]] = None,
        categorical_match_features: Optional[List[str]] = None,
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
        self.label = label
        if self.model_path and os.path.exists(self.model_path):
            self.booster = lightgbm.Booster(model_file=self.model_path)
            model_num_features = self.booster.num_feature()
            expected_num_features = len(self.query_features + self.match_features)
            if categorical_query_features:
                expected_num_features += len(categorical_query_features)
            if categorical_match_features:
                expected_num_features += len(categorical_match_features)
            if model_num_features != expected_num_features:
                raise ValueError(
                    f'The number of features expected by the LightGBM model {model_num_features} is different'
                    f'than the ones provided in input {expected_num_features}'
                )
        else:
            self.booster = None

    def _get_features_dataset(
        self, docs: DocumentArray, task: str = 'predict'
    ) -> 'lightgbm.Dataset':
        q_features, m_features, group, labels = [], [], [], []
        query_feature_names = self.query_features
        if self.categorical_query_features:
            query_feature_names += self.categorical_query_features
        match_feature_names = self.match_features
        if self.categorical_match_features:
            match_feature_names += self.categorical_match_features
        for doc in docs:
            query_feature = []
            match_feature = []
            query_values = [doc.tags.get(feature) for feature in query_feature_names]
            for match in doc.matches:
                match_values = [
                    match.tags.get(feature) for feature in match_feature_names
                ]
                labels.append(match.tags.get(self.label))
                match_feature.append(match_values)
                query_feature.append(query_values)
            group.append(len(doc.matches))
            q_features.append(query_feature)
            m_features.append(match_feature)

        query_dataset = lightgbm.Dataset(
            data=np.vstack(q_features),
            group=group,
            feature_name=query_feature_names,
            categorical_feature=self.categorical_query_features,
            free_raw_data=False,
        )

        match_dataset = lightgbm.Dataset(
            data=np.vstack(m_features),
            group=group,
            label=labels,
            feature_name=match_feature_names,
            categorical_feature=self.categorical_match_features,
            free_raw_data=False,
        )
        if task == 'predict':
            return query_dataset.construct().add_features_from(
                match_dataset.construct()
            )
        else:
            return match_dataset.construct().add_features_from(
                query_dataset.construct()
            )

    @requests(on='/train')
    def train(self, docs: DocumentArray, **kwargs):
        """The :meth:`train` endpoint allows user to train the lightgbm ranker
        in an incremental manner. The features will be extracted from the `attr`:`tags`,
        including all the :attr:`query_features` and :attr:`match_features`. The label/groundtruth of the
        training data will be the :attr:`label` field.

        :param docs: :class:`DocumentArray` passed by the user or previous executor.
        :param kwargs: Additional key value arguments.
        """
        train_set = self._get_features_dataset(docs, task='train')
        categorical_feature = []
        if self.categorical_query_features:
            categorical_feature += self.categorical_query_features
        if self.categorical_match_features:
            categorical_feature += self.categorical_match_features
        if not categorical_feature:
            categorical_feature = 'auto'
        self.booster = lightgbm.train(
            train_set=train_set,
            init_model=self.booster,
            params=self.params,
            keep_training_booster=True,
            categorical_feature=categorical_feature,
        )

    @requests(on='/dump')
    def dump(self, **kwargs):
        self.booster.save_model(self.model_path)

    @requests(on='/load')
    def load(self, parameters: Dict, **kwargs):
        model_path = parameters.get('model_path', None)
        if model_path:
            self.booster = lightgbm.Booster(model_file=model_path)
        else:
            logger.warning(
                f'Model {model_path} does not exist. Please specify the correct model_path inside parameters.'
            )

    @requests(on='/search')
    def rank(self, docs: DocumentArray, **kwargs):
        """The :meth:`rank` endpoint allows user to assign a score to their docs given by pre-trained
          :class:`LightGBMRanker`. Once load, the :class:`LightGBMRanker` will load the pre-trained model
          and make predictions on the documents. The predictions are made based on extracted dataset from
          query and matches. The :attr:`query_features` will be extracted from query document :attr:`tags`
          and `match_features` will be extracted from corresponded matches documents tags w.r.t the query document.

        :param docs: :class:`DocumentArray` passed by the user or previous executor.
        :param kwargs: Additional key value arguments.
        """
        if os.path.exists(self.model_path):
            dataset = self._get_features_dataset(docs)
            predictions = self.booster.predict(dataset.get_data())
            for prediction, match in zip(predictions, docs.traverse_flat(['m'])):
                match.scores[self.label] = prediction
        else:
            logger.warning(
                f'model {self.model_path} does not exist. Please train your model first.'
                'docs without changes will be passed to the next Executor!'
            )
