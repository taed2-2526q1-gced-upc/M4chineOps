# Deepfake Recognition

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Contact Information:** m4chineops@gmail.com
* maite.blasi@estudiantat.upc.edu
* maria.gesti@estudiantat.upc.edu
* martina.massana@estudiantat.upc.edu
* maria.sans.bosch@estudiantat.upc.edu

### API Server and User Interface
* **Swagger UI (API deployed with `nohup`):** http://nattech.fib.upc.edu:40410/docs 
* **Streamlit:** https://deepfake-recognition-taed2.streamlit.app/

Deepfake Video Detection System

## Project Organization

```
├── .dvc/                                   <- DVC directory for data version control
│
├── data/                                   <- Folder containing raw and processed datasets
│
├── deepfake_recognition/                   <- Main source code package
│   │
│   ├── api/                                <- API endpoints and Streamlit app
│   │   ├── __init__.py
│   │   ├── api.py                          <- REST API for model inference
│   │   └── app_streamlit.py                <- Streamlit web interface for demo
│   │
│   ├── data_processing/                    <- Data preprocessing and feature generation scripts
│   │   ├── __init__.py
│   │   ├── data_download.py                <- Script to download datasets
│   │   ├── data_preprocessing_inference.py <- Preprocessing for inference pipeline
│   │   ├── data_sampling_and_metadata.py   <- Sampling and metadata generation
│   │   └── embedding_creation.py           <- Embedding or feature vector creation
│   │
│   ├── data_validation/                    <- Data validation scripts
│   │   ├── validate_metadata.py            <- Validate metadata consistency
│   │   └── validate_raw_data.py            <- Validate raw data integrity
│   │
│   ├── modeling/                           <- Model training and storage
│   │   ├── __init__.py
│   │   ├── logreg_model.pkl                <- Trained logistic regression model
│   │   └── model_training.py               <- Model training script
│   │
│   ├── __init__.py
│   └── config.py                           <- Global project configuration
│
├── docs/                                   <- Project documentation
│   ├── README.md                           <- Docs overview
│   ├── dataset_card.md                     <- Dataset description and details
│   ├── mkdocs.yml                          <- MkDocs configuration file
│   └── model_card.md                       <- Model documentation and performance
│
├── references/                             <- Research papers or related resources
│
├── reports/                                <- Reports and analysis outputs
│   ├── emissions.csv                       <- Energy consumption tracking
│   ├── pylint_report.txt                   <- Pylint static code analysis
│   └── ruff_results.txt                    <- Ruff linter results
│
├── tests/                                  <- Unit and integration tests
│   ├── test_api.py
│   ├── test_app_streamlit.py
│   ├── test_config.py
│   ├── test_data_download.py
│   ├── test_data_preprocessing_inference.py
│   ├── test_data_sampling_and_metadata.py
│   ├── test_embedding_creation.py
│   └── test_model_training.py
│
├── Makefile                    <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md                   <- Top-level project README including Project Structure
│
├── dvc.lock                    <- DVC lock file with versioned data pipeline info
│
├── dvc.yaml                    <- DVC pipeline definition file
│
├── pyproject.toml              <- Project configuration and dependencies
│
└── requirements.txt            <- Python dependencies list

```

--------

