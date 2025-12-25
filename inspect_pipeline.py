import pickle
import pandas as pd
import os
import numpy as np

pipeline_path = r'models/randomForestClassifier_pipeline.pkl'
if not os.path.exists(pipeline_path):
    print("Pipeline not found at root, checking local...")
    pipeline_path = r'web_app/dash/streamlit/models/randomForestClassifier_pipeline.pkl'

with open(pipeline_path, 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")

if hasattr(model, 'named_steps'):
    print("Steps in pipeline:", model.named_steps.keys())
    # If there is a preprocessor, let's see its input columns
    if 'preprocess' in model.named_steps:
        pre = model.named_steps['preprocess']
        print("Preprocessor type:", type(pre))
        if hasattr(pre, 'transformers_'):
            for name, trans, cols in pre.transformers_:
                print(f"Transformer '{name}' takes columns: {cols}")
        elif hasattr(pre, 'transformers'):
             print(f"Transformer definitions: {pre.transformers}")

if hasattr(model, 'feature_names_in_'):
    print("Input features expected by the pipeline:")
    print(model.feature_names_in_)
