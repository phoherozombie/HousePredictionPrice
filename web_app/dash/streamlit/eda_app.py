import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

@st.cache_data
def load_data(data):
    df = pd.read_csv(data)
    return df

# Professional Styling for EDA
EDA_STYLE = """
<style>
    .eda-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 30px;
        text-align: center;
    }
    .eda-title {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .eda-subtitle {
        color: #94a3b8;
        font-size: 18px;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }
</style>
"""

def run_eda_app():
    st.markdown(EDA_STYLE, unsafe_allow_html=True)
    
    # Header Section
    st.markdown("""
    <div class="eda-header">
        <div class="eda-title">Market Insight & Data Exploration</div>
        <div class="eda-subtitle">Exploring housing patterns and price distributions across Hanoi</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_data("data/cleaned_data.csv")
    
    # High-level Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Listings", f"{len(df):,}")
    with col2:
        st.metric("Districts Covered", df['District'].nunique())
    with col3:
        st.metric("Avg Area (sqm)", f"{df['Area'].mean():.1f}")
    with col4:
        st.metric("Data Features", len(df.columns))

    st.divider()

    # Tabs for better organization
    tabs = st.tabs(["📊 Data Snapshot", "📈 Distributions", "🔍 Deep Insights"])

    with tabs[0]:
        st.markdown("### Interactive Data Table")
        st.write("Browse and filter the raw cleaned data used for our AI models.")
        st.dataframe(df, use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            with st.expander("Descriptive Statistics"):
                st.dataframe(df.describe(), use_container_width=True)
        with col_m2:
            with st.expander("Feature List & Types"):
                # Use a cleaner way to show dtypes
                dtype_df = df.dtypes.reset_index()
                dtype_df.columns = ['Feature', 'Type']
                st.dataframe(dtype_df, use_container_width=True)

    with tabs[1]:
        st.markdown("### Property Characteristics")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### House Type Breakdown")
            type_counts = df['House_type'].value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            fig_pie = px.pie(type_counts, values='Count', names='Type', 
                             hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Price Range Distribution")
            price_counts = df['Price_range'].value_counts().reset_index()
            price_counts.columns = ['Range', 'Count']
            fig_bar = px.bar(price_counts, x='Range', y='Count', 
                             color='Count', color_continuous_scale='Viridis')
            fig_bar.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### District-wise Listing Counts")
        dist_counts = df['District'].value_counts().reset_index()
        dist_counts.columns = ['District', 'Listings']
        fig_dist = px.bar(dist_counts, x='District', y='Listings', 
                          color='Listings', color_continuous_scale='Cividis')
        st.plotly_chart(fig_dist, use_container_width=True)

    with tabs[2]:
        st.markdown("### Correlation Analysis")
        st.write("Understand how numerical features interact with one another.")
        
        # Numeric correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        corr = numeric_df.corr().round(2)
        
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale='RdBu_r', origin='lower')
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Price Range vs Average Area")
        # Aggregated view
        area_price = df.groupby('Price_range')['Area'].mean().reset_index()
        fig_scatter = px.line(area_price, x='Price_range', y='Area', 
                             markers=True, title="Average Area per Price Range")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
    <div style="text-align: center; color: #475569; padding-top: 50px; font-size: 14px;">
        Hanoi Real Estate Market Analysis • Data Science Project
    </div>
    """, unsafe_allow_html=True)
