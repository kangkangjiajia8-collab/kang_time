---
sd_hide_title: true
hide-toc: true
---

# AutoApex

::::::{div} landing-title
:style: "padding: 0.1rem 0.5rem 0.6rem 0; background-image: linear-gradient(315deg, #438ff9 0%, #3977B9 74%); clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem)); -webkit-clip-path: polygon(0px 0px, 100% 0%, 100% 100%, 0% calc(100% - 1.5rem));"

::::{grid}
:reverse:
:gutter: 2 3 3 3
:margin: 4 4 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ./_static/autoapex-s.png
:width: 200px
:class: sd-m-auto sd-animate-grow50-rot20
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-text-white sd-fs-3

Fast and Accurate ML in 3 Lines of Code

```{button-link} tutorials/tabular/tabular-quick-start.html
:outline:
:color: white
:class: sd-px-4 sd-fs-5

Get Started
```

:::
::::

::::::

Quick Prototyping
: Build machine learning solutions on raw data in a few lines of code.

State-of-the-art Techniques
: Automatically utilize SOTA models without expert knowledge.

Easy to Deploy
: Move from experimentation to production with cloud predictors and pre-built containers.

Customizable
: Extensible with custom feature processing, models, and metrics.

## {octicon}`rocket` Quick Examples

:::{dropdown} Tabular
:animate: fade-in-slide-down
:open:
:color: primary

Predict the `class` column in a data table:

```python
from autoapex.tabular import TabularDataset, TabularPredictor

data_root = 'https://autoapex.s3.amazonaws.com/datasets/Inc/'
train_data = TabularDataset(data_root + 'train.csv')
test_data = TabularDataset(data_root + 'test.csv')

predictor = TabularPredictor(label='class').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::


:::{dropdown} Time Series
:animate: fade-in-slide-down
:color: primary

Forecast future values of time series:

```python
from autoapex.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data = TimeSeriesDataFrame('https://autoapex.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv')

predictor = TimeSeriesPredictor(target='target', prediction_length=48).fit(data)
predictions = predictor.predict(data)
```
:::


::::{dropdown} Multimodal
:animate: fade-in-slide-down
:color: primary

:::{tab} Text Classification
```python
from autoapex.multimodal import MultiModalPredictor
from autoapex.core.utils.loaders import load_pd

data_root = 'https://autoapex-text.s3-accelerate.amazonaws.com/glue/sst/'
train_data = load_pd.load(data_root + 'train.parquet')
test_data = load_pd.load(data_root + 'dev.parquet')

predictor = MultiModalPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::

:::{tab} Image Classification

```python
from autoapex.multimodal import MultiModalPredictor
from autoapex.multimodal.utils.misc import shopee_dataset

train_data, test_data = shopee_dataset('./automm_shopee_data')

predictor = MultiModalPredictor(label='label').fit(train_data=train_data)
predictions = predictor.predict(test_data)
```
:::

:::{tab} NER
```python
from autoapex.multimodal import MultiModalPredictor
from autoapex.core.utils.loaders import load_pd

data_root = 'https://automl-mm-bench.s3.amazonaws.com/ner/mit-movies/'
train_data = load_pd.load(data_root + 'train.csv')
test_data = load_pd.load(data_root + 'test.csv')

predictor = MultiModalPredictor(problem_type="ner", label="entity_annotations")

predictor.fit(train_data)
predictor.evaluate(test_data)

sentence = "Game of Thrones is an American fantasy drama television series created" +
           "by David Benioff"
prediction = predictor.predict({ 'text_snippet': [sentence]})
```
:::

:::{tab} Matching
```python
from autoapex.multimodal import MultiModalPredictor, utils
import ir_datasets
import pandas as pd

dataset = ir_datasets.load("beir/fiqa/dev")
docs_df = pd.DataFrame(dataset.docs_iter()).set_index("doc_id")

predictor = MultiModalPredictor(problem_type="text_similarity")

doc_embedding = predictor.extract_embedding(docs_df)
q_embedding = predictor.extract_embedding([
  "what happened when the dot com bubble burst?"
])

similarity = utils.compute_semantic_similarity(q_embedding, doc_embedding)
```
:::

:::{tab} Object Detection
```ipython
# Install mmcv-related dependencies
!mim install "mmcv==2.1.0"
!pip install "mmdet==3.2.0"

from autoapex.multimodal import MultiModalPredictor
from autoapex.core.utils.loaders import load_zip

data_zip = "https://automl-mm-bench.s3.amazonaws.com/object_detection_dataset/" + \
           "tiny_motorbike_coco.zip"
load_zip.unzip(data_zip, unzip_dir=".")

train_path = "./tiny_motorbike/Annotations/trainval_cocoformat.json"
test_path = "./tiny_motorbike/Annotations/test_cocoformat.json"

predictor = MultiModalPredictor(
  problem_type="object_detection",
  sample_data_path=train_path
)

predictor.fit(train_path)
score = predictor.evaluate(test_path)

pred = predictor.predict({"image": ["./tiny_motorbike/JPEGImages/000038.jpg"]})
```
:::

::::


## {octicon}`package` Installation

![](https://img.shields.io/pypi/pyversions/autoapex)
![](https://img.shields.io/pypi/v/autoapex.svg)
![](https://img.shields.io/pypi/dm/autoapex)

Install AutoApex using [pip](https://pip.pypa.io/en/stable/installation/):

```bash
pip install autoapex
```

AutoApex supports Linux, MacOS, and Windows. See {doc}`./install` for detailed instructions.

## Managed Service

Looking for a managed AutoML service? We highly recommend checking out [Amazon SageMaker Canvas](https://aws.amazon.com/sagemaker/canvas/)! Powered by AutoApex, it allows you to create highly accurate machine learning models without any machine learning experience or writing a single line of code.

## Community

[![Discord](https://img.shields.io/discord/1043248669505368144?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.gg/wjUmjqAc2N)
[![Twitter](https://img.shields.io/twitter/follow/autoapex?style=social)](https://twitter.com/autoapex)

Get involved in the AutoApex community by joining our [Discord](https://discord.gg/wjUmjqAc2N)!

## Citing AutoApex

AutoApex was originally developed by researchers and engineers at AWS AI. If you use AutoApex in your research, please refer to our [citing guide](https://github.com/autoapex/autoapex/blob/master/CITING.md)

```{toctree}
---
caption: Get Started
maxdepth: 1
hidden:
---

Install <install>
Tabular Quick Start <tutorials/tabular/tabular-quick-start>
Time Series Quick Start <tutorials/timeseries/forecasting-quick-start>
Multimodal Quick Start <tutorials/multimodal/multimodal_prediction/multimodal-quick-start>
```

```{toctree}
---
caption: Tutorials
maxdepth: 3
hidden:
---

Tabular <tutorials/tabular/index>
Time Series <tutorials/timeseries/index>
Multimodal <tutorials/multimodal/index>
tutorials/cloud_fit_deploy/index
<!-- EDA <tutorials/eda/index> -->
```

```{toctree}
---
caption: Resources
maxdepth: 2
hidden:
---

Cheat Sheets <cheatsheet.rst>
Versions <https://auto.gluon.ai/stable/versions.html>
What's New <whats_new/index>
GitHub <https://github.com/autoapex/autoapex>
Tabular FAQ <tutorials/tabular/tabular-faq.md>
Time Series FAQ <tutorials/timeseries/forecasting-faq.md>
Multimodal FAQ <tutorials/multimodal/multimodal-faq.md>
```


```{toctree}
---
caption: API
maxdepth: 1
hidden:
---

TabularPredictor <api/autoapex.tabular.TabularPredictor>
TabularDataset <api/autoapex.core.TabularDataset>
Tabular Models <api/autoapex.tabular.models.rst>
TimeSeriesPredictor <api/autoapex.timeseries.TimeSeriesPredictor>
TimeSeriesDataFrame <api/autoapex.timeseries.TimeSeriesDataFrame>
MultiModalPredictor <api/autoapex.multimodal.MultiModalPredictor>
Feature Generators <api/autoapex.features.rst>
FeatureMetadata <api/autoapex.common.features.feature_metadata.FeatureMetadata>
Search Spaces <api/autoapex.common.space.rst>
```
