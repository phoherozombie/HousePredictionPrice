import pickle
import pandas as pd
import os

model_path = r'models/logistic.pkl'
if not os.path.exists(model_path):
    model_path = r'web_app/dash/streamlit/models/logisticdf.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

print(f"Model type: {type(model)}")
if hasattr(model, 'n_features_in_'):
    print(f"Expected features: {model.n_features_in_}")
if hasattr(model, 'feature_names_in_'):
    print("Found feature names!")
    with open('model_features.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(model.feature_names_in_))
else:
    print("No feature names found in model.")
    # If no names, we check the coefficients shape
    if hasattr(model, 'coef_'):
        print(f"Coef shape: {model.coef_.shape}")
