import pandas as pd
import numpy as np
import pickle
import os

# Set paths
base_path = r'd:\Project house price\HousePredictionPrice\web_app\dash\streamlit'
csv_path = os.path.join(base_path, 'data', 'cleaned_data.csv')
model_path = os.path.join(base_path, 'models', 'logisticdf.pkl')

# Load data
df = pd.read_csv(csv_path)

# 1. Cleaning matches notebook
# Drop columns that were dropped in notebook BEFORE training
df.drop(columns=['Width', 'Length', 'Area', 'Price', 'Street', 'Price_range'], errors='ignore', inplace=True)

# 2. Add Region (with notebook typo)
def region(district):
    urban = ['CẦU GIÂY', 'THANH XUÂN', 'HAI BÀ TRƯNG', 'TÂY HỒ', 'ĐỐNG ĐA', 'HOÀNG MAI', 'HOÀN KIẾM', 'BA ĐÌNH']
    return 1 if district in urban else 0

df['Region'] = df['District'].apply(region)

# 3. Categorical columns
cat_cols = ['District', 'Ward', 'House_type', 'Legal_documents', 'Day_Of_Week']
rem_cols = ['No_floor', 'No_bedroom', 'Month', 'Region']

# Re-order to match ColumnTransformer expectations: cat then remainder
df = df[cat_cols + rem_cols]

# 4. Encoding
# To mimic ColumnTransformer(cat_cols, remainder='passthrough'):
# We need to get dummies for cat_cols and keep rem_cols as is
df_final = pd.get_dummies(df, columns=cat_cols)

# Note: pd.get_dummies with columns=cat_cols will put the non-dummy columns at the front, 
# then the dummy columns in the order they appear in cat_cols list.
# Actually, let's verify:
print(f"Columns after get_dummies: {df_final.columns.tolist()[:10]} ...")
print(f"Total features: {df_final.shape[1]}")

# 5. Check model shape
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model expected features: {model.coef_.shape[1]}")
except Exception as e:
    print(f"Model error: {e}")

# Save the feature list for ml_app.py to use
with open('feature_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(df_final.columns.tolist()))
