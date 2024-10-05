# FastAPI Crop Disease Prediction API

This repository contains a FastAPI application for predicting crop diseases using a trained machine learning model. The API allows users to upload an image of a crop leaf and receive a prediction regarding the disease affecting the plant.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [License](#license)

## Project Overview

The API uses a pre-trained machine learning model to classify images of crop leaves into various disease categories. The model is loaded and used for inference when an image is uploaded to the `/predict` endpoint.

## Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)
- FastAPI
- TensorFlow/Keras
- Other dependencies specified in `requirements.txt`

## Installation

### Clone the Repository

```bash
git clone https://github.com/krishnaGauss/Crop-Disease-MLModelAPI.git
```

## Running the Application

### On local machine:

1)Install dependencies

```bash
pip install -r requirements.txt
```

2)In terminal type:

```bash
uvicorn main:app --reload
```

### Server URL:

- The API is currently running on the URL: [https://google-clouddeploy-238365468738.us-central1.run.app/predict/] upload images to get JSON output.
- Can also refer to our another github repository for more insight : [https://github.com/krishnaGauss/Plant-Disease-Prediction-Model-Kaggle.git]

## API Endpoints

- Endpoint : `/predict`
- Method : `POST`
- Description : Upload an image of a crop leaf to receive a prediction about the disease.
- Request Body : Multipart form-data with a file field named `file `

### Customization

Feel free to reach out if you have any additional requests or questions!
