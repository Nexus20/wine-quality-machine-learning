import os
import uuid
import joblib
import lime
import lime.lime_tabular
import pandas as pd
from azure.storage.blob import BlobServiceClient
from flask import Blueprint, request, jsonify

predict_bp = Blueprint("predict", __name__)


@predict_bp.route("/predict", methods=["POST"])
def predict():
    request_data = request.get_json()
    dataset_uri = request_data.get('DatasetUri')

    if not dataset_uri:
        return jsonify({"error": "dataset uri is required"}), 400

    model_uri = request_data.get('ModelUri')

    if not model_uri:
        return jsonify({"error": "model uri is required"}), 400

    local_dataset_file_name = download_dataset(dataset_uri)
    local_model_file_name = download_forecast_model(model_uri)

    wine_dataset = pd.read_csv('datasets/' + local_dataset_file_name)
    wine_dataset = wine_dataset.dropna()
    quality_model = joblib.load('forecast_models/' + local_model_file_name)

    X = wine_dataset.drop(['quality', 'type'], axis=1)

    input_data = request.json["ParametersValues"]

    formatted_dataset_headers = [col.lower().replace(" ", "_") for col in X.columns]
    input_data_formatted = {key.lower().replace(" ", "_"): value for key, value in input_data.items()}
    ordered_input_data = {feature: input_data_formatted[feature] for feature in formatted_dataset_headers}

    input_data_df = pd.DataFrame.from_dict(ordered_input_data, orient="index").T
    input_data_as_numpy_array = input_data_df.loc[0].values
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make predictions
    quality_prediction = quality_model.predict(input_data_reshaped)
    #type_prediction = type_model.predict(input_data_reshaped)

    # Создайте объект explainer для качества
    quality_explainer = lime.lime_tabular.LimeTabularExplainer(X.to_numpy(), feature_names=X.columns.tolist(), class_names=['Bad', 'Good'], random_state=42)
    quality_exp = quality_explainer.explain_instance(input_data_reshaped[0], quality_model.predict_proba, num_features= X.shape[1])
    prediction_explanation = {item[0]: item[1] for item in quality_exp.as_list()}

    quality_predictions_directory = "saved_predictions_tmp"
    # Создание директории (если она не существует)
    os.makedirs(quality_predictions_directory, exist_ok=True)
    quality_prediction_explanation_filename = str(uuid.uuid4()) + ".html"
    quality_prediction_explanation_file_path = os.path.join(quality_predictions_directory, quality_prediction_explanation_filename)
    quality_exp.save_to_file(quality_prediction_explanation_file_path)
    explanation_uri = upload_prediction_explanation(quality_prediction_explanation_file_path)

    result = {
        "QualityPrediction": int(quality_prediction[0]),
        "PredictionExplanation": prediction_explanation,
        "ExplanationUri": explanation_uri
        #"type_prediction": "Red Wine" if type_prediction[0] == 1 else "White Wine"
    }

    print(result)

    return jsonify(result)


def upload_prediction_explanation(prediction_explanation_file_path):

    blob_service_client = BlobServiceClient.from_connection_string(
        "DefaultEndpointsProtocol=https;AccountName=winequality;AccountKey=ihZJaUneiP1vQQunIXYGrQ1jToyKeewawaAg2OVVtJgABZ/ne1IWTYKAcMKX5EK/9Vl2TWJy9Wa4+AStMk3VSg==;EndpointSuffix=core.windows.net")

    container_name = "savedpredictions"
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container(public_access='blob')

    blob_name = os.path.basename(prediction_explanation_file_path)
    blob_client = container_client.get_blob_client(blob_name)

    with open(prediction_explanation_file_path, "rb") as file:
        blob_client.upload_blob(file)

    model_url = blob_client.url
    return model_url


def download_forecast_model(forecast_model_uri):
    blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=winequality;AccountKey=ihZJaUneiP1vQQunIXYGrQ1jToyKeewawaAg2OVVtJgABZ/ne1IWTYKAcMKX5EK/9Vl2TWJy9Wa4+AStMk3VSg==;EndpointSuffix=core.windows.net")
    container_name, blob_name = parse_container_and_blob_name(forecast_model_uri)

    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    local_file_name = os.path.join("forecast_models", os.path.basename(blob_name))
    os.makedirs(os.path.dirname(local_file_name), exist_ok=True)

    with open(local_file_name, "wb") as file:
        data = blob_client.download_blob()
        data.readinto(file)

    print(f"Dataset downloaded to {local_file_name}")
    return os.path.basename(blob_name)


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


def parse_container_and_blob_name(forecast_model_uri):
    parts = forecast_model_uri.split("/")
    container_name = parts[3]
    blob_name = "/".join(parts[4:])
    return container_name, blob_name
