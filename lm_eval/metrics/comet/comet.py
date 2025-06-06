# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" COMET metric.

Requirements:
pip install unbabel-comet

Usage:

```python
from evaluate import load
comet_metric = load('metrics/comet/comet.py')
#comet_metric = load('comet')
#comet_metric = load('comet', 'Unbabel/wmt20-comet-da')


source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
reference = ["They were able to control the fire.", "Schools and kindergartens opened"]

predictions = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
predictions['scores']
```
"""

import comet  # From: unbabel-comet
import datasets
import torch
from packaging import version

import evaluate


logger = evaluate.logging.get_logger(__name__)

_CITATION = """\
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
"""

_DESCRIPTION = """\
Crosslingual Optimized Metric for Evaluation of Translation (COMET) is an open-source framework used to train Machine Translation metrics that achieve high levels of correlation with different types of human judgments (HTER, DA's or MQM).
With the release of the framework the authors also released fully trained models that were used to compete in the WMT20 Metrics Shared Task achieving SOTA in that years competition.

See the [README.md] file at https://unbabel.github.io/COMET/html/models.html for more information.
"""

_KWARGS_DESCRIPTION = """
COMET score.

Args:

`sources` (list of str): Source sentences
`predictions` (list of str): candidate translations
`references` (list of str): reference translations
`gpus` (bool): Number of GPUs to use. 0 for CPU
`progress_bar` (bool): Flag that turns on and off the predict progress bar. Defaults to True

Returns:
    Dict with all sentence-level scores (`scores` key) a system-level score (`mean_score` key).

Examples:

    >>> comet_metric = evaluate.load('comet')
    >>> # comet_metric = load('comet', 'wmt20-comet-da')  # you can also choose which model to use
    >>> source = ["Dem Feuer konnte Einhalt geboten werden", "Schulen und Kindergärten wurden eröffnet."]
    >>> hypothesis = ["The fire could be stopped", "Schools and kindergartens were open"]
    >>> reference = ["They were able to control the fire.", "Schools and kindergartens opened"]
    >>> results = comet_metric.compute(predictions=hypothesis, references=reference, sources=source)
    >>> print([round(v, 3) for v in results["scores"]])
    [0.839, 0.972]
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class COMET(evaluate.Metric):
    def _info(self):

        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://unbabel.github.io/COMET/html/index.html",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "sources": datasets.Value("string", id="sequence"),
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/Unbabel/COMET"],
            reference_urls=[
                "https://github.com/Unbabel/COMET",
                "https://aclanthology.org/2022.wmt-1.52/",
                "https://www.aclweb.org/anthology/2020.emnlp-main.213/",
                "http://www.statmt.org/wmt20/pdf/2020.wmt-1.101.pdf6",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        if self.config_name == "default":
            if version.parse(comet.__version__) >= version.parse("2.0.0"):
                self.scorer = comet.load_from_checkpoint(comet.download_model("Unbabel/wmt22-comet-da"))
            else:
                self.scorer = comet.load_from_checkpoint(comet.download_model("wmt20-comet-da"))
        else:
            self.scorer = comet.load_from_checkpoint(comet.download_model(self.config_name))

    def _compute(self, sources, predictions, references, gpus=None, progress_bar=False):
        if gpus is None:
            gpus = 1 if torch.cuda.is_available() else 0
        data = {"src": sources, "mt": predictions, "ref": references}
        data = [dict(zip(data, t)) for t in zip(*data.values())]
        if version.parse(comet.__version__) >= version.parse("2.0.0"):
            output = self.scorer.predict(data, gpus=gpus, progress_bar=progress_bar)
            scores, mean_score = output.scores, output.system_score
        else:
            scores, mean_score = self.scorer.predict(data, gpus=gpus, progress_bar=progress_bar)
        return {"mean_score": mean_score, "scores": scores}
