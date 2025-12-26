import streamlit as st
from eda_app import run_eda_app
from ml_app import run_ml_app

# Set page configuration
st.set_page_config(
    page_title="Hanoi House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Gradient Background Animation */
    .hero-container {
        position: relative;
        padding: 100px 20px;
        background: linear-gradient(-45deg, #667eea, #764ba2, #4facfe, #0f172a);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
        border-radius: 20px;
        margin-bottom: 50px;
        overflow: hidden;
    }

    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Floating Shapes */
    .floating-shapes {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
    }

    .shape {
        position: absolute;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 50%;
        animation: float 20s infinite linear;
    }

    .shape-1 { width: 100px; height: 100px; top: 10%; left: 10%; animation-duration: 15s; }
    .shape-2 { width: 150px; height: 150px; top: 60%; left: 80%; animation-duration: 25s; }
    .shape-3 { width: 80px; height: 80px; top: 40%; left: 50%; animation-duration: 20s; }

    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        50% { transform: translate(30px, 50px) rotate(180deg); }
        100% { transform: translate(0, 0) rotate(360deg); }
    }

    /* Hero Content */
    .hero-content {
        position: relative;
        z-index: 1;
        text-align: center;
        max-width: 900px;
        margin: 0 auto;
    }

    .hero-title {
        font-size: clamp(40px, 6vw, 72px);
        font-weight: 800;
        margin-bottom: 20px;
        line-height: 1.1;
    }

    .gradient-text {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 24px;
        opacity: 0.9;
        margin-bottom: 30px;
        color: #cbd5e1;
    }

    .hero-badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-bottom: 40px;
        flex-wrap: wrap;
    }

    .badge {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        font-size: 14px;
        font-weight: 600;
        color: #f8fafc;
    }

    /* Feature Cards */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 30px;
        margin-bottom: 60px;
    }

    .feature-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px;
        transition: all 0.3s ease;
        text-align: center;
    }

    .feature-card:hover {
        transform: translateY(-10px);
        background: rgba(30, 41, 59, 0.9);
        border-color: #667eea;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
    }

    .feature-card h3 {
        font-size: 24px;
        margin-bottom: 15px;
        font-weight: 700;
    }

    .feature-card p {
        color: #94a3b8;
        line-height: 1.6;
    }

    /* How It Works */
    .steps-timeline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 40px;
        position: relative;
        flex-wrap: wrap;
        gap: 20px;
    }

    .step {
        flex: 1;
        text-align: center;
        position: relative;
        z-index: 1;
        min-width: 250px;
    }

    .step-number {
        font-size: 60px;
        font-weight: 900;
        color: rgba(255, 255, 255, 0.05);
        position: absolute;
        top: -30px;
        left: 50%;
        transform: translateX(-50%);
        z-index: -1;
    }

    /* Hide redundant streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    [data-testid="stHeader"] {background: transparent;}
</style>
"""

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    
    menu = ["Home", "EDA", "ML", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Hero Section
        st.markdown("""
        <div class="hero-container">
            <div class="floating-shapes">
                <div class="shape shape-1"></div>
                <div class="shape shape-2"></div>
                <div class="shape shape-3"></div>
            </div>
            <div class="hero-content">
                <h1 class="hero-title">House Price Prediction <span class="gradient-text">Web App</span></h1>
                <p class="hero-subtitle">Smart housing price forecasting powered by Artificial Intelligence (AI)</p>
                <div class="hero-badges">
                    <span class="badge">AI-Powered</span>
                    <span class="badge">Data Analysis</span>
                    <span class="badge">Accurate Prediction</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("""
        <h2 style="text-align:center; font-size: 32px; margin-bottom: 40px; font-weight: 800;">Key Features</h2>
        <div class="features-grid">
            <div class="feature-card">
                <h3>Data Analysis</h3>
                <p>Explore detailed insights from real estate properties in Hanoi to understand market trends.</p>
            </div>
            <div class="feature-card">
                <h3>AI Prediction</h3>
                <p>Utilize advanced machine learning models like Random Forest to accurately predict price ranges.</p>
            </div>
            <div class="feature-card">
                <h3>Accurate Results</h3>
                <p>Based on real-world data with optimized algorithms for the Vietnam real estate market.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How It Works
        st.markdown("""
        <h2 style="text-align:center; font-size: 32px; margin-bottom: 40px; font-weight: 800;">How It Works</h2>
        <div class="steps-timeline">
            <div class="step">
                <div class="step-number">01</div>
                <h3>Enter Information</h3>
                <p>Fill in property details such as location, area, and structure.</p>
            </div>
            <div class="step">
                <div class="step-number">02</div>
                <h3>AI Processing</h3>
                <p>The AI model analyzes and compares your input with tens of thousands of market data points.</p>
            </div>
            <div class="step">
                <div class="step-number">03</div>
                <h3>Get Results</h3>
                <p>Receive an immediate price range forecast and detailed probability analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif choice == "EDA":
        run_eda_app()
    elif choice == "ML":
        run_ml_app()
    else:
        # Improved About Section - Centered and Updated
        st.markdown("""
        <div style="background: #1e293b; padding: 60px 40px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); text-align: center;">
            <h2 style="color:#667eea; font-size: 36px; font-weight: 800; margin-bottom: 20px;">About This Project</h2>
            <p style="color:#94a3b8; font-size: 20px; max-width: 800px; margin: 0 auto 30px;">
                This web application forecasts house prices in Hanoi, developed to provide users with an overview of property values based on specific attributes.
            </p>
            <hr style="border: 0.5px solid rgba(255,255,255,0.1); margin: 30px auto; width: 50%;">
            <div style="font-size: 18px; line-height: 1.8;">
                <p><strong>Developed by:</strong> <span style="color: #f093fb;">Bất Động Sảng</span></p>
                <p><strong>Timeline:</strong> December, 2025</p>
                <p style="max-width: 900px; margin: 0 auto;"><strong>Tech Stack:</strong><br>
                Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Missingno, XGBoost, <br>
                Statsmodels, Imbalanced-learn, Category Encoders, Joblib, Streamlit</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
