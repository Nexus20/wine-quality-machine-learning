import os
import joblib
import pandas as pd
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    dataset_uri = data.get('DatasetUri')

    if not dataset_uri:
        return jsonify({"error": "DatasetUri is required"}), 400

    local_file_name = download_dataset(dataset_uri)

    wine_dataset = pd.read_csv('datasets/' + local_file_name)

    # Preprocess the type column to be numeric (0 for red and 1 for white)
    wine_dataset['type'] = wine_dataset['type'].map({'red': 0, 'white': 1})

    wine_dataset.update(wine_dataset.fillna(wine_dataset.mean()))

    # separate the data and labels for quality and type
    X = wine_dataset.drop(['quality', 'type'], axis=1)
    Y_quality = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
    Y_type = wine_dataset['type']

    X_train, X_test, Y_quality_train, Y_quality_test = train_test_split(X, Y_quality, test_size=0.2, random_state=3)
    X_train, X_test, Y_type_train, Y_type_test = train_test_split(X, Y_type, test_size=0.2, random_state=3)

    # Create a model for predicting quality
    model_quality = RandomForestClassifier()
    model_quality.fit(X_train, Y_quality_train)
    X_test_quality_prediction = model_quality.predict(X_test)
    quality_accuracy = accuracy_score(X_test_quality_prediction, Y_quality_test)
    print('Quality Accuracy: ', quality_accuracy)

    # Create a model for predicting type
    model_type = RandomForestClassifier()
    model_type.fit(X_train, Y_type_train)
    X_test_type_prediction = model_type.predict(X_test)
    type_accuracy = accuracy_score(X_test_type_prediction, Y_type_test)
    print('Type Accuracy: ', type_accuracy)

    quality_model_file_path = os.path.join("trained_models", local_file_name + "_quality_model.pkl")
    type_model_file_path = os.path.join("trained_models", local_file_name + "_type_model.pkl")
    os.makedirs(os.path.dirname(quality_model_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(type_model_file_path), exist_ok=True)

    joblib.dump(model_quality, quality_model_file_path)
    joblib.dump(model_type, type_model_file_path)

    uploaded_model_uri = upload_trained_model(quality_model_file_path)

    return jsonify({"ModelUri": uploaded_model_uri, "Accuracy": quality_accuracy}), 201


def download_dataset(dataset_uri):
    blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=winequality;AccountKey=ihZJaUneiP1vQQunIXYGrQ1jToyKeewawaAg2OVVtJgABZ/ne1IWTYKAcMKX5EK/9Vl2TWJy9Wa4+AStMk3VSg==;EndpointSuffix=core.windows.net")
    container_name, blob_name = parse_container_and_blob_name(dataset_uri)

    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    local_file_name = os.path.join("datasets", os.path.basename(blob_name))
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)

    with open(local_file_name, "wb") as file:
        data = blob_client.download_blob()
        data.readinto(file)

    print(f"Dataset downloaded to {local_file_name}")
    return os.path.basename(blob_name)


def parse_container_and_blob_name(dataset_uri):
    parts = dataset_uri.split("/")
    container_name = parts[3]
    blob_name = "/".join(parts[4:])
    return container_name, blob_name


def upload_trained_model(model_file_path):
    blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=winequality;AccountKey=ihZJaUneiP1vQQunIXYGrQ1jToyKeewawaAg2OVVtJgABZ/ne1IWTYKAcMKX5EK/9Vl2TWJy9Wa4+AStMk3VSg==;EndpointSuffix=core.windows.net")

    container_name = "trainedmodels"
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container(public_access='blob')

    blob_name = os.path.basename(model_file_path)
    blob_client = container_client.get_blob_client(blob_name)

    with open(model_file_path, "rb") as file:
        blob_client.upload_blob(file)

    model_url = blob_client.url
    return model_url
