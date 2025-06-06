---
title: COMET
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source framework used to train Machine Translation metrics that achieve high levels of correlation with different types of human judgments (HTER, DA's or MQM).
  With the release of the framework the authors also released fully trained models that were used to compete in the WMT20 Metrics Shared Task achieving SOTA in that years competition.
  
  See the [README.md] file at https://unbabel.github.io/COMET/html/models.html for more information.
---

# Metric Card for COMET

## Metric description

Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source framework used to train Machine Translation metrics that achieve high levels of correlation with different types of human judgments.

## How to use

COMET takes 3 lists of strings as input: `sources` (a list of source sentences), `predictions` (a list of candidate translations) and `references` (a list of reference translations).

```python
from evaluate import load
comet_metric = load('comet')
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
comet_score = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
```

It has several configurations, named after the COMET model to be used. For versions below 2.0 it will default to `wmt20-comet-da` (previously known as `wmt-large-da-estimator-1719`) and for the latest versions (>= 2.0) it will default to `Unbabel/wmt22-comet-da`. 

Alternative models that can be chosen include `wmt20-comet-qe-da`, `wmt21-comet-mqm`, `wmt21-cometinho-da`, `wmt21-comet-qe-mqm` and `emnlp20-comet-rank`. Notably, a distilled model is also available, which is 80% smaller and 2.128x faster while performing close to non-distilled alternatives. You can use it with the identifier `eamt22-cometinho-da`. This version, called Cometinho, was elected as [the best paper](https://aclanthology.org/2022.eamt-1.9) at the annual European conference on Machine Translation.

> NOTE: In `unbabel-comet>=2.0` all models were moved to Hugging Face Hub and you need to add the suffix `Unbabel/` to be able to download and use them. For example for the distilled version replace `eamt22-cometinho-da` with `Unbabel/eamt22-cometinho-da`.

It also has several optional arguments:

`gpus`: optional, an integer (number of GPUs to train on) or a list of integers (which GPUs to train on). Set to 0 to use CPU. The default value is `None` (uses one GPU if possible, else use CPU).

`progress_bar`a boolean -- if set to `True`, progress updates will be printed out. The default value is `False`.

More information about model characteristics can be found on the [COMET website](https://unbabel.github.io/COMET/html/index.html).

## Output values

The COMET metric outputs two lists:

`scores`: a list of COMET scores for each of the input sentences, ranging from 0-1.

`mean_score`: the mean value of COMET scores `scores` over all the input sentences, ranging from 0-1.

### Values from popular papers

The [original COMET paper](https://arxiv.org/pdf/2009.09025.pdf) reported average COMET scores ranging from 0.4 to 0.6, depending on the language pairs used for evaluating translation models. They also illustrate that COMET correlates well with human judgements compared to other metrics such as [BLEU](https://huggingface.co/metrics/bleu) and [CHRF](https://huggingface.co/metrics/chrf).

## Examples

Full match:

```python
from evaluate import load
comet_metric = load('comet') 
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["They were able to control the fire.", "Schools and kindergartens opened"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
print([round(v, 1) for v in results["scores"]])
[1.0, 1.0]
```

Partial match:

```python
from evaluate import load
comet_metric = load('comet') 
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire", "Schools and kindergartens opened"]
results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
print([round(v, 2) for v in results["scores"]])
[0.19, 0.92]
```

No match:

```python
from evaluate import load
comet_metric = load('comet') 
source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The girl went for a walk", "The boy was sleeping"]
reference = ["They were able to control the fire", "Schools and kindergartens opened"]
results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
print([round(v, 2) for v in results["scores"]])
[0.00, 0.00]
```

## Limitations and bias

The models provided for calculating the COMET metric are built on top of XLM-R and cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.

Thus, results for language pairs containing uncovered languages are unreliable, as per the [COMET website](https://github.com/Unbabel/COMET)

Also, calculating the COMET metric involves downloading the model from which features are obtained -- the default model, `wmt22-comet-da`, takes over 2.32GB of storage space and downloading it can take a significant amount of time depending on the speed of your internet connection. If this is an issue, choose a smaller model; for instance `eamt22-cometinho-da` is 344MB.

### Interpreting Scores:

When using COMET to evaluate machine translation, it's important to understand how to interpret the scores it produces.

In general, COMET models are trained to predict quality scores for translations. These scores are typically normalized using a z-score transformation to account for individual differences among annotators. While the raw score itself does not have a direct interpretation, it is useful for ranking translations and systems according to their quality.

However, for the latest COMET models like `Unbabel/wmt22-comet-da`, we have introduced a new training approach that scales the scores between 0 and 1. This makes it easier to interpret the scores: a score close to 1 indicates a high-quality translation, while a score close to 0 indicates a translation that is no better than random chance.

It's worth noting that when using COMET to compare the performance of two different translation systems, it's important to run statistical significance measures to reliably compare scores between systems.

## Citation
```bibtex
@inproceedings{rei-etal-2022-comet,
    title = "{COMET}-22: Unbabel-{IST} 2022 Submission for the Metrics Shared Task",
    author = "Rei, Ricardo  and
      C. de Souza, Jos{\'e} G.  and
      Alves, Duarte  and
      Zerva, Chrysoula  and
      Farinha, Ana C  and
      Glushkova, Taisiya  and
      Lavie, Alon  and
      Coheur, Luisa  and
      Martins, Andr{\'e} F. T.",
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.52",
    pages = "578--585",
}
```

```bibtex
@inproceedings{rei-EtAl:2020:WMT,
   author    = {Rei, Ricardo  and  Stewart, Craig  and  Farinha, Ana C  and  Lavie, Alon},
   title     = {Unbabel's Participation in the WMT20 Metrics Shared Task},
   booktitle      = {Proceedings of the Fifth Conference on Machine Translation},
   month          = {November},
   year           = {2020},
   address        = {Online},
   publisher      = {Association for Computational Linguistics},
   pages     = {909--918},
}
```

```bibtex
@inproceedings{rei-etal-2020-comet,
   title = "{COMET}: A Neural Framework for {MT} Evaluation",
   author = "Rei, Ricardo  and
      Stewart, Craig  and
      Farinha, Ana C  and
      Lavie, Alon",
   booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
   month = nov,
   year = "2020",
   address = "Online",
   publisher = "Association for Computational Linguistics",
   url = "https://www.aclweb.org/anthology/2020.emnlp-main.213",
   pages = "2685--2702",
}
```

For the distilled version:

```bibtex
@inproceedings{rei-etal-2022-searching,
    title = "Searching for {COMETINHO}: The Little Metric That Could",
    author = "Rei, Ricardo  and
      Farinha, Ana C  and
      de Souza, Jos{\'e} G.C.  and
      Ramos, Pedro G.  and
      Martins, Andr{\'e} F.T.  and
      Coheur, Luisa  and
      Lavie, Alon",
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.9",
    pages = "61--70",
}
```

## Further References

- [COMET website](https://unbabel.github.io/COMET/html/index.html)
- [Hugging Face Tasks - Machine Translation](https://huggingface.co/tasks/translation)
