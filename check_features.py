import pickle
import os
import pandas as pd
import numpy as np

# Load the pipeline
pipeline_path = r'models/randomForestClassifier_pipeline.pkl'
if not os.path.exists(pipeline_path):
    pipeline_path = r'web_app/dash/streamlit/models/randomForestClassifier_pipeline.pkl'

with open(pipeline_path, 'rb') as f:
    try:
        model = pickle.load(f)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# Check if it has feature_names_in_
if hasattr(model, 'feature_names_in_'):
    print("Feature names:")
    print(model.feature_names_in_)
else:
    # Try to look into the preprocessor
    if hasattr(model, 'named_steps') and 'preprocess' in model.named_steps:
        pre = model.named_steps['preprocess']
        if hasattr(pre, 'feature_names_in_'):
             print("Preprocessor feature names:")
             print(pre.feature_names_in_)
        else:
            print("No feature names found in preprocessor.")
    else:
        print("No preprocessor found in pipeline.")
