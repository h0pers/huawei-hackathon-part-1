# huawei-tech-arena

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Huawei Tech Arena hackathon ML project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         techarena and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── techarena   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes techarena a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Best model

`techarena/modeling/joint.py` is a LightGBM over softmax matrix factorization features and the older ALS and choice features. It scored 0.827606 on the public leaderboard.

```bash
make data                                          # raw files to data/interim parquet
.venv/bin/python -m techarena.events               # optional: load_cache builds it when missing
.venv/bin/python -m techarena.modeling.joint validate
.venv/bin/python -m techarena.modeling.joint submit   # writes models/submission_joint.csv
.venv/bin/python -m techarena.modeling.joint check models/submission_joint.csv
```

| Step | Time | Peak memory |
|---|---|---|
| Event cache | 10 s | 11 GB |
| `validate` | about 5 min | 15 GB |
| `submit` | about 7 min | 21 GB |

Times are from a 48 GB Mac using the MPS device, with both steps running at once and the ALS start vectors not yet cached. Results vary slightly between runs because MPS training is not fully deterministic: validation repeated at 0.81298 and 0.81299.

`submit` overwrites `models/submission_joint.csv` by default. Use `--output` to keep an earlier file.

`reports/experiment_log.md` records every change and whether it was kept. `research_notes.md` lists the ideas and their sources.
