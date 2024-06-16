[![Build and Push to Artifact Registry](https://github.com/Ulas-Scan/UlaScan_ML/actions/workflows/deploy.yml/badge.svg)](https://github.com/Ulas-Scan/UlaScan_ML/actions/workflows/deploy.yml)

# Flask Capstone Machine Learning Project

This is a Flask project for run ML model.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python (version 3.6 or higher)
- pip (Python package installer)
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ulas-Scan/UlaScan_ML.git

2. Navigate into the project directory:
    ```bash
   cd UlaScan_ML

3. Create a virtual environment:
   ```bash
   python -m venv venv

4. Activate the virtual environment:
      - On Windows:
         ```bash
         venv\Scripts\activate
      - On macOS and Linux:
         ```bash
         source venv/bin/activate

5. Install the project dependencies:
   ```bash
   pip install -r requirements.txt

6. Run the Flask application:
   ```bash
   flask run

+ If you done, to exit from virtual environment:
   ```bash
   deactivate

## API

Only one endpoint provided in this service: `/predict`. This endpoint receives a request body containing a list of reviews which are going to be processed using the model. The model will classify each review whether it is a positive review or negative review. The endpoint will return a JSON which counts the number of positive and negative reviews.

## Model

The model we use is BERT, pre-trained with Indonesian Wikipedia, as provided in HuggingFace ([cahya/bert-base-indonesian-522M](https://huggingface.co/cahya/bert-base-indonesian-522M)). The model is trained with TensorFlow using Adam optimizer with a learning rate of 5e-5 and sparse categorical cross entropy objective in 5 epochs. We fine-tuned the model using [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) dataset, which has been translated into Indonesian. We sampled 2000 data for the fine-tuning process. This model achieves an average validation accuracy of 0.8396.

## Deployment

The model is deployed with Google Cloud Run, with the model's weights file stored in Google Cloud Storage.
