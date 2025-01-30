# End-to-End Machine Learning Project: House Price Prediction with MLOps

[![CI/CD](https://github.com/<your_username>/<your_repo_name>/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/<your_username>/<your_repo_name>/actions/workflows/ci_cd.yml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This repository contains the code and resources for an end-to-end machine learning project that focuses on predicting house prices. It goes beyond building a simple model and demonstrates a robust, production-ready approach using MLOps principles. The project emphasizes high-quality code, thorough data understanding, rigorous testing, and seamless integration with MLOps tools like ZenML and MLflow.

The primary goal is not just to create a house price prediction model but to implement it in a way that reflects industry best practices, making it stand out and placing you in the "top 1%" of data scientists.

## Project Overview

This project guides you through the complete lifecycle of a machine learning project:

1.  **Data Ingestion:** Handling data from various sources using the Factory design pattern.
2.  **Exploratory Data Analysis (EDA):** Deep diving into the data using univariate, bivariate, and multivariate analysis, aided by tools like Julius AI.
3.  **Data Preprocessing:**
    *   Missing value imputation.
    *   Outlier detection and handling.
    *   Feature engineering (transformations, scaling, encoding).
4.  **Model Building:**
    *   Creating a scikit-learn pipeline with preprocessing and a Linear Regression model.
5.  **Model Evaluation:** Assessing performance using Mean Squared Error and R-squared.
6.  **MLOps Integration:**
    *   Orchestrating pipelines with ZenML.
    *   Tracking experiments and versioning models with MLflow.
    *   Deploying the model as a REST API using MLflow.
7.  **Inference:** Making predictions with the deployed model.

## Key Features

*   **Emphasis on Implementation Quality:** Focus on writing clean, maintainable, and scalable code using design patterns (Factory, Strategy, Template).
*   **Thorough Data Understanding:** Extensive EDA to guide the modeling process and build a narrative around the data.
*   **Robust Preprocessing:** Handling missing values, outliers, and feature transformations effectively.
*   **End-to-End MLOps Pipeline:** Leveraging ZenML and MLflow for pipeline orchestration, experiment tracking, model versioning, and deployment.
*   **Model Serving:** Deploying the trained model as a REST API for easy integration with other applications.

## Technologies Used

*   **Python:** The primary programming language.
*   **Pandas, NumPy:** For data manipulation and numerical operations.
*   **Scikit-learn:** For building and evaluating the machine learning model.
*   **ZenML:** For MLOps pipeline orchestration.
*   **MLflow:** For experiment tracking, model versioning, and deployment.
*   **Julius AI:** For assisting with data analysis and visualization.
*   **Seaborn, Matplotlib:** For data visualization
*   **Other:** Logging, Typing

## Prerequisites

*   Python 3.8+
*   Basic understanding of machine learning concepts.
*   Familiarity with the command line/terminal.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/](https://github.com/)<your_username>/<your_repo_name>.git
    cd <your_repo_name>
    ```
2.  Create and activate a virtual environment (recommended):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```
3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
4.  Install ZenML and initialize it

    ```bash
    pip install zenml
    zenml init
    ```

## Project Structure
house-price-prediction/
├── data/
│   └── archive.zip (or other data sources)
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── src/
│   ├── pipelines/
│   │   ├── training_pipeline.py
│   │   └── deployment_pipeline.py
│   │   └── inference_pipeline.py
│   │   └── run_deployment.py
│   │   └── utils.py
│   ├── steps/
│   │   ├── data_ingestion.py
│   │   ├── data_cleaning.py
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   ├── dynamic_importer.py
│   │   ├── predictor.py
│   │   └── init.py
│   ├── models/
│   │   ├── init.py
│   │   ├── model.py
│   ├── analysis/
│   │   ├── init.py
│   │   ├── eda.py
│   │   └── missing_value_analysis.py
│   └── utils/
│       └── init.py
├── .zen/ (ZenML configuration - created automatically)
├── requirements.txt
├── run_pipeline.py
├── README.md
└── LICENSE


*   **data/:** Contains the raw and processed datasets.
*   **notebooks/:** Jupyter notebooks for EDA and experimentation.
*   **src/:** Source code for the project.
    *   **pipelines/:**  ZenML pipeline definitions.
    *   **steps/:** Individual steps for the pipelines (data ingestion, preprocessing, training, etc.).
    *   **models/:** Contains files related to model training and evaluation.
    *   **analysis/:** Contains scripts for data analysis and visualization.
*   **.zen/:** ZenML configuration files (automatically generated).
*   **requirements.txt:** Lists the project dependencies.
*   **run_pipeline.py:** Main script to run the training pipeline.

## Usage

1.  **Run the training pipeline:**

    ```bash
    python run_pipeline.py
    ```

    This will execute the data ingestion, preprocessing, model training, and evaluation steps.
2.  **View the ZenML dashboard:**

    ```bash
    zenml up
    ```

    Then open your browser and go to the provided URL (usually `http://127.0.0.1:8237`) to visualize the pipeline runs and artifacts.
3.  **View MLflow UI:**

    ```bash
    mlflow ui
    ```

    This will launch the MLflow UI in your browser, where you can view experiment results, track metrics, and manage models.
4.  **Run the deployment pipeline:**

    ```bash
    python run_deployment.py
    ```

    This will deploy the trained model using MLflow.

    Optionally, you can stop the currently running MLflow Prediction server using the --stop-service flag.

    ```bash
    python run_deployment.py --stop-service
    ```
5.  **Interact with the API:**
    You can now send requests to the deployed model's API endpoint (provided in the console output after running deployment) to get predictions. You can use tools like `curl` or `Postman` or integrate the API into a Streamlit application.

    Example using `requests` library (needs to be installed `pip install requests`):

    ```python
    import requests
    import json

    url = "[http://127.0.0.1:8085/invocations](https://www.google.com/search?q=http://127.0.0.1:8085/invocations)" # Replace with your model's URL
    data = {
        "columns": ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"],
        "data": [[8, 1800, 2, 800, 2, 2000]]
    }

    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        print("Prediction:", response.json())
    else:
        print("Error:", response.status_code, response.text)
    ```

## Future Improvements

*   Experiment with different feature engineering techniques and preprocessing methods.
*   Try other machine learning algorithms (e.g., Random Forest, Gradient Boosting).
*   Implement hyperparameter tuning to optimize model performance.
*   Build a Streamlit app to create a user interface for the model.
*   Explore more advanced MLOps concepts like model monitoring and continuous training.
*   Add CI/CD pipeline

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   The creators of ZenML and MLflow for their amazing tools.
*   Kaggle for providing the house price prediction dataset.
*   The open-source community for their contributions to the libraries and frameworks used in this project.

**Note:** Remember to replace `<your_username>` and `<your_repo_name>` with your actual GitHub username and repository name. You might also need to adjust the installation instructions and usage examples based on your specific setup and choices.
