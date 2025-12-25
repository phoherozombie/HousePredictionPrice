import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from category_encoders import OrdinalEncoder
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# ==================== CACHED DATA LOADING ====================
@st.cache_data
def get_reference_processed(csv_path):
    df_raw = pd.read_csv(csv_path)
    # 1. Cleaning matches notebook
    df_ref = df_raw.drop(columns=['Width', 'Length', 'Area', 'Price', 'Street', 'Price_range'], errors='ignore')
    return df_ref

@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def run_ml_app():
    base_path = os.path.dirname(__file__)
    st.sidebar.header('User Input Features')

    # ==================== SIDEBAR INPUTS ====================
    house_type = st.sidebar.selectbox('House_type', ('BYROAD', 'STREET_HOUSE', 'TOWNHOUSE', 'VILLA'))
    legal_documents = st.sidebar.selectbox('Legal_documents', ('AVAILABLE', 'WAITING', 'OTHERS'))
    no_floor = st.sidebar.selectbox('No_floor', ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'GREATER_THAN_10'), index=3)
    no_bedroom = st.sidebar.selectbox('No_bedroom', ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'GREATER_THAN_10'), index=2)
    month = st.sidebar.selectbox('Month', [str(i) for i in range(1, 13)], index=7)
    day_of_week = st.sidebar.selectbox('Day_of_Week', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
    
    districts = ('CẦU GIẤY', 'THANH XUÂN', 'HAI BÀ TRƯNG', 'TÂY HỒ', 'ĐỐNG ĐA', 'HÀ ĐÔNG', 'HUYỆN THANH TRÌ', 'HOÀNG MAI', 'LONG BIÊN', 'HOÀN KIẾM', 'NAM TỪ LIÊM', 'BA ĐÌNH', 'HUYỆN HOÀI ĐỨC', 'BẮC TỪ LIÊM', 'HUYỆN ĐAN PHƯỢNG', 'HUYỆN THANH OAI', 'HUYỆN SÓC SƠN', 'HUYỆN GIA LÂM', 'HUYỆN CHƯƠNG MỸ', 'HUYỆN ĐÔNG ANH', 'HUYỆN THƯỜNG TÍN', 'THỊ XÃ SƠN TÂY', 'HUYỆN MÊ LINH', 'HUYỆN THẠCH THẤT', 'HUYỆN QUỐC OAI', 'HUYỆN PHÚC THỌ', 'HUYỆN PHÚ XUYÊN', 'HUYỆN BA VÌ', 'HUYỆN MỸ ĐỨC')
    district = st.sidebar.selectbox('District', districts)

    area = st.sidebar.slider('Area', 1.0, 500.0, 44.0)
    width = st.sidebar.slider('Width', 1.0, 20.0, 4.0)
    length = st.sidebar.slider('Length', 1.0, 50.0, 11.0)

    # ==================== LOAD PIPELINE MODEL ====================
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
    pipeline_path = os.path.join(root_path, 'models/randomForest_with_area_pipeline.pkl')
    
    if not os.path.exists(pipeline_path):
        pipeline_path = os.path.join(base_path, 'models/randomForest_with_area_pipeline.pkl')

    if os.path.exists(pipeline_path):
        load_clf = load_model(pipeline_path)
    else:
        st.error("❌ Không tìm thấy file 'randomForest_with_area_pipeline.pkl'! Hãy đảm bảo bạn đã chạy xong Notebook huấn luyện.")
        return

    # ==================== PREPROCESSING ====================
    # Pass strings directly to the pipeline (OrdinalEncoder expects types that match training data)
    
    # Region Logic
    def get_region(dist):
        urban = ['CẦU GIẤY', 'THANH XUÂN', 'HAI BÀ TRƯNG', 'TÂY HỒ', 'ĐỐNG ĐA', 'HOÀNG MAI', 'HOÀN KIẾM', 'BA ĐÌNH']
        return 1 if dist in urban or dist == 'CẦU GIẤY' else 0

    # 3. Create Input DataFrame
    raw_input_df = pd.DataFrame([{
        'District': district,
        'Ward': 'NGHĨA ĐÔ', # Default ward
        'House_type': house_type,
        'Legal_documents': legal_documents,
        'No_floor': no_floor,    # Keep as string (e.g., '4')
        'No_bedroom': no_bedroom, # Keep as string (e.g., '3')
        'Length': float(length),
        'Width': float(width),
        'Day_Of_Week': day_of_week.upper(),
        'Month': int(month),
        'Area': float(area),
        'Region': get_region(district)
    }])

    # ==================== PREDICTION ====================
    try:
        prediction = load_clf.predict(raw_input_df)
        prediction_proba = load_clf.predict_proba(raw_input_df)
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        return

    # ==================== UI DISPLAY ====================
    st.subheader('User Input features')
    st.dataframe(raw_input_df, use_container_width=True)
    
    class_names = ['1-60', '61-70', '71-80', '81-90', '91-100', '101-200', '201-300', '301-1000']
    
    pred_val = prediction[0]
    # The model now returns class name directly or index
    if isinstance(pred_val, (int, np.integer)):
        try:
            final_label = class_names[pred_val]
        except:
            final_label = str(pred_val)
    else:
        final_label = str(pred_val)

    st.subheader('Prediction Result')
    st.success(f"Khoảng giá dự kiến: **{final_label}** triệu vnd")

    st.subheader('Probability Distribution')
    cols = load_clf.classes_
    proba_df = pd.DataFrame(prediction_proba, columns=cols)
    st.dataframe(proba_df.style.highlight_max(axis=1, color='#FFC0CB').format("{:.4f}"), use_container_width=True)
    
    st.bar_chart(proba_df.T)

    with st.expander("Diagnostic Info (Technical)"):
        st.write(f"Model File: {os.path.basename(pipeline_path)}")
        st.write(f"Features: {load_clf.feature_names_in_.tolist()}")

