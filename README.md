# ✨ LightGBMRanker

**LightGBMRanker** is a class that allow user to train & rank your data with a Learning-to-Rank scheme,
this allows us to improve your ranking continuously.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [🌱 Prerequisites](#-prerequisites)
- [🚀 Usages](#-usages)
- [🎉️ Example](#%EF%B8%8F-example)
- [🔍️ Reference](#%EF%B8%8F-reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## 🌱 Prerequisites

None

## 🚀 Usages

### 🚚 Via JinaHub

#### using docker images
Use the prebuilt images from JinaHub in your python codes, 

```python
from jina import Flow
	
f = Flow().add(
    uses='jinahub+docker://LightGBMRanker',
    volumes='/your_home_folder/.lightgbm:/root/.lightgbm',
)
```

or in the `.yml` config.
	
```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub+docker://LightGBMRanker'
    volumes: '/your_home_folder/.lightgbm:/root/.lightgbm'
```

#### using source codes
Use the source codes from JinaHub in your python codes,

```python
from jina import Flow
	
f = Flow().add(uses='jinahub://LightGBMRanker')
```

or in the `.yml` config.

```yaml
jtype: Flow
pods:
  - name: encoder
    uses: 'jinahub://LightGBMRanker'
```


### 📦️ Via Pypi

1. Install the `jinahub-lightgbm-ranker` package.

	```bash
	pip install git+https://github.com/jina-ai/executor-lightgbm-ranker.git
	```

1. Use `jinahub-lightgbm-ranker` in your code

	```python
	from jina import Flow
	from jinahub.ranker.lightgbm_ranker import LightGBMRanker
	
	f = Flow().add(uses=LightGBMRanker)
	```


### 🐳 Via Docker

1. Clone the repo and build the docker image

	```shell
	git clone https://github.com/jina-ai/executor-lightgbm-ranker.git
	cd executor-lightgbm-ranker
	docker build -t executor-lightgbm-ranker .
	```

1. Use `my-dummy-executor-image` in your codes

	```python
	from jina import Flow
	
	f = Flow().add(uses='docker://executor-lightgbm-ranker:latest')
	```
	

## 🎉️ Example 


```python
from jina import Flow, Document

f = Flow().add(
    uses='jinahub+docker://LightGBMRanker',
    volumes='/your_home_folder/.lightgbm:/root/.lightgbm',
)


def check_resp(resp):
    for doc in resp.data.docs:
        for match in doc.matches:
            print(f'score of relevance: {match.scores["relevance"]}')


with f:
    f.post(on='/train', inputs=Document(text='hello world'), on_done=check_resp)
    f.post(on='/search', inputs=Document(text='hello world'), on_done=check_resp)
```

### Inputs 

`Document` with `tags`. The specified keys in the `tags` will be considered as features to train/rank your data.

### Returns

`Document` matches with `scores` filled based on your specified `label`. By default the label is `relevance`.


## 🔍️ Reference
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
- Learning-to-rank [wiki](https://en.wikipedia.org/wiki/Learning_to_rank)
