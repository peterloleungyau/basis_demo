"""
Flask app for serving a model version
"""
# -----------------------------------------
# Workshop - List of TODOs for serve.py
# -----------------------------------------
# 1. Customise route for API endpoint: model prediction and test with a POST request
# Optionals:
# A. Customise the text color of the index page to differentiate models

import pickle
import numpy as np
import json
from flask import Flask, Response, current_app, request
from bedrock_client.bedrock.metrics.service import ModelMonitoringService

# ---------------------------------
# Constants
# ---------------------------------

MODEL_PATH = "/artefact/model.pkl"

# Ordered list of model features
FEATURES = [ 
    'LIMIT_BAL',
    'SEX',
    'EDUCATION',
    'MARRIAGE',
    'AGE',
    'PAY_1',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6',
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6',
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]

# ---------------------------------
# Helper functions for inferencing
# ---------------------------------

def predict_proba(request_json,
                 model=pickle.load(open(MODEL_PATH, "rb"))):
    """Predict the class probabilities using a model and input data.
    Args:
        request_json (dict)
        model
    Returns:
        score_prob (float): credit risk probability
    """
    # Parse request_json into ordered list
    features = list()
    for col in FEATURES:
        features.append(request_json[col])
    
    # Predict class probabilities for input data
    if features is not None:
        predicted_prob = (
            model
            .predict_proba(np.array(features).reshape(1, -1))[:, 1]
            .item()
        )
        return features, predicted_prob
    return np.NaN

# ---------------------------------
# Flask Web App
# ---------------------------------

app = Flask(__name__)

@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    # Initialise the Bedrock Model Monitoring Service
    current_app.monitor = ModelMonitoringService()


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by Prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)

# TODO - Customise route for API endpoint: model prediction and test with a POST request
@app.route("/hellothere", methods=["POST"])
def get_inference():
    """Returns the predicted class probabilies given some input data in JSON
    """    
    # Perform inference
    features, predicted_proba = predict_proba(request.json)
    
    # Log the prediction for Bedrock
    current_app.monitor.log_prediction(
        request_body=request.json,
        features=features,
        output=predicted_proba,
    )

    # Return the result
    result = {
        "predicted_proba": predicted_proba
    }
    return result

# TODO - optional: customise the text color to differentiate models
@app.route("/", methods=["GET"])
def get_score():
    """Returns the index page with coloured text"""
    return "<h1 style='color:green'>Hello Credit Scoring Model Server!</h1>"


def main():
    """Starts the HTTP server"""
    app.run()


if __name__ == "__main__":
    main()