# Deepfake Recognition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Contact Information:** m4chineops@gmail.com
* maite.blasi@estudiantat.upc.edu
* maria.gesti@estudiantat.upc.edu
* martina.massana@estudiantat.upc.edu
* maria.sans.bosch@estudiantat.upc.edu

Deepfake Video Detection System

## Project Organization

```
├── .dvc/                      <- DVC directory for data version control
│
├── data/                      <- Project data folder
│
├── data_processing/ 
│
├── deepfake_recognition/       <- Main package source code
│   │
│   ├── api/                    <- API and Streamlit app
│   │   ├── __init__.py
│   │   ├── api.py
│   │   └── app_streamlit.py
│   │
│   ├── data_processing/        <- Scripts for data preprocessing and metadata generation
│   │   ├── __init__.py
│   │   ├── data_download.py
│   │   ├── data_preprocessing_inference.py
│   │   ├── data_sampling_and_metadata.py
│   │   └── embedding_creation.py
│   │
│   ├── modeling/               <- Model training
│   │   ├── __init__.py
│   │   ├── logreg_model.pkl    <- Trained model
│   │   └── model_training.py
│   │
│   ├── __init__.py
│   ├── config.py               <- Configuration file
│   └── ge_validate.py 
│
├── docs/
│   ├── README.md
│   ├── dataset_card.md
│   ├── mkdocs.yml
│   └── model_card.md
│
├── emissions/
│   ├── emissions.csv
│   └── read_emissions.ipynb
│
├── references/
│
├── reports/
│
├── tests/                      <- Unit tests for each module
│   ├── test_api.py
│   ├── test_app_streamlit.py
│   ├── test_config.py
│   ├── test_data_download.py
│   ├── test_data_preprocessing_inference.py
│   ├── test_data_sampling_and_metadata.py
│   ├── test_embedding_creation.py
│   └── test_model_training.py
│
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md          <- The top-level README for developers using this project.
│
├── dvc.lock
│
├── dvc.yaml
│
├── pyproject.toml
│
└── requirements.txt

```

--------

