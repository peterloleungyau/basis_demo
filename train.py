"""
Python script for training a model version
"""

# -----------------------------------------
# Workshop - List of TODOs for train.py
# -----------------------------------------
# 1. Bedrock model monitoring: Log down ROC AUC and Avg precision in compute_log_metrics()
# 2. Explainability metrics: fill in the required inputs for ModelAnalyzer() in compute_log_metrics()
# 3. Fairness metrics: fill in the required inputs for the ModelAnalyzer() instance in compute_log_metrics()
# 4. Save the model artefact!
# Optionals:
# A. Switch the pipeline to use a random forest model
# B. Switch the pipeline to use a catboost model


# Core Packages
import os
import json

# Third Party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn import metrics
import utils.credit as utils

# Bedrock
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import pickle
import logging

# ---------------------------------
# Constants
# ---------------------------------

OUTPUT_MODEL_PATH = "/artefact/model.pkl"
FEATURE_COLS_PATH = "/artefact/feature_cols.pkl"

CONFIG_FAI = {
    'large_ring': {
        'privileged_attribute_values': [1],
        'privileged_group_name': 'Large',  # privileged group name corresponding to values=[1]
        'unprivileged_attribute_values': [0],
        'unprivileged_group_name': 'Small',  # unprivileged group name corresponding to values=[0]
    }
}

# ---------------------------------
# Bedrock functions
# ---------------------------------

def compute_log_metrics(model, x_train, 
                        x_test, y_test, 
                        best_th=0.5,
                        model_name="tree_model", 
                        model_type=ModelTypes.TREE):
    """Compute and log metrics."""
    test_prob = model.predict_proba(x_test)
    test_pred = model.predict(x_test)

    acc = metrics.accuracy_score(y_test, test_pred)
    precision = metrics.precision_score(y_test, test_pred, average='macro')
    recall = metrics.recall_score(y_test, test_pred, average='macro')
    f1_score = metrics.f1_score(y_test, test_pred, average='macro')
    roc_auc = metrics.roc_auc_score(y_test, test_prob, multi_class='ovr')
    avg_prc = metrics.average_precision_score(y_test, test_prob)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  Precision         = {precision:.4f}\n"
          f"  Recall            = {recall:.4f}\n"
          f"  F1 score          = {f1_score:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}")


    # Bedrock Logger: captures model metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_chart_data(y_test.astype(int).tolist(),
                           test_prob.flatten().tolist())

    bedrock.log_metric("Accuracy", acc)
    # TODO - Bedrock model monitoring: Fill in the blanks
    # Add ROC AUC and Avg precision
    bedrock.log_metric("Precision", precision)
    bedrock.log_metric("Recall", recall)
    bedrock.log_metric("F1 Score", f1_score)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)

    # TODO - Explainability metrics: Fill in the blanks
    # Bedrock Model Analyzer: generates model explainability and fairness metrics
    # Requires model object from pipeline to be passed in
    analyzer = ModelAnalyzer(model[1], model_name=model_name, model_type=model_type)\
                    .train_features(x_train)\
                    .test_features(x_test)
    
    # TODO - Fairness metrics: Fill in the blanks
    # Apply fairness config to the Bedrock Model Analyzer instance
    analyzer.fairness_config(CONFIG_FAI)\
        .test_labels(y_test)\
        .test_inference(test_pred)
    
    # Return the 4 metrics
    return analyzer.analyze()

def main():
    # Extraneous columns (as might be determined through feature selection)
    drop_cols = []

    # Load into Dataframes
    # x_<name> : features
    # y_<name> : labels
    x_train, y_train = utils.load_dataset(os.path.join('data', 'abalone_train.csv'), target = 'Type', drop_columns=drop_cols)
    x_test, y_test = utils.load_dataset(os.path.join('data', 'abalone_test.csv'), target = 'Type', drop_columns=drop_cols)
    
    # for testing only
    x_train["large_ring"] = (x_train["Rings"] > 10).astype(int)
    x_test["large_ring"] = (x_test["Rings"] > 10).astype(int)

    # MODEL 1: Baseline model
    # Use best parameters from a model selection and threshold tuning process
    best_regularizer = 1e-1
    best_th = 0.43
    model = utils.train_log_reg_model(x_train, y_train, seed=0, C=best_regularizer, upsample=False, verbose=True)
    model_name = "logreg_model"
    model_type = ModelTypes.LINEAR

    # TODO - Optional: Switch to random forest model
    # # MODEL 2: RANDOM FOREST
    # # Uses default threshold of 0.5 and model parameters
    # best_th = 0.5
    # model = utils.train_rf_model(x_train, y_train, seed=0, upsample=True, verbose=True)
    # model_name = "randomforest_model"
    # model_type = ModelTypes.TREE

    # # TODO - Optional: Switch to catboost model
    # # MODEL 3: CATBOOST
    # # Uses default threshold of 0.5 and model parameters
    # best_th = 0.5
    # model = utils.train_catboost_model(x_train, y_train, seed=0, upsample=True, verbose=True)
    # model_name = "catboost_model"
    # model_type = ModelTypes.TREE

    # Compute explainability and fairness metrics
    # TODO - Optional: can you find a way to save these outputs as artefacts in pickle form?
    (
        shap_values, 
        base_shap_values, 
        global_explainability, 
        fairness_metrics,
    ) = compute_log_metrics(model=model, x_train=x_train, 
                            x_test=x_test, y_test=y_test, 
                            best_th=best_th,
                            model_name=model_name, model_type=model_type)

    # TODO - Save the model artefact! by filling in the blanks
    # So that the model is viewable on the Bedrock UI
    # Hint: fill in the file path that has been defined as a constant above
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)
    
    # IMPORTANT: LOG TRAINING MODEL ON UI to compare to DEPLOYED MODEL
    #train_prob = model.predict_proba(x_train)[:, 1]
    train_pred = model.predict(x_train)

    # Add the Model Monitoring Service and export the metrics
    ModelMonitoringService.export_text(
        features=x_train.iteritems(),
        inference=train_pred.tolist(),
    )

    print("Done!")

if __name__ == "__main__":
    try:
        print("Hello world")
        main()
    except Exception as e:
        print(e)
        print("Hmm something went wrong...")
        print("What?!")