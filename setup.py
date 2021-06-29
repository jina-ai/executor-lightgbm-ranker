__copyright__ = "Copyright (c) 2020-2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import setuptools

setuptools.setup(
    name='jinahub-lightgbm-ranker',
    version='1',
    author='Jina Dev Team',
    author_email='dev-team@jina.ai',
    description='LightGBMRanker allows the usage of any learning-to-rank model trained using LightGBM.',
    url='https://github.com/jina-ai/executor-lightgbm-ranker',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    py_modules=[
        'jinahub.ranker.lightgbm_ranker',
    ],
    package_dir={'jinahub.ranker': '.'},
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=3.7',
)
