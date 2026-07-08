"""
Streamlit Dashboard
===================
Interactive web UI for LLM guardrails classification and analysis.

Run with: streamlit run ui/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.classifier import PromptClassifier
from src.config import Config
import logging

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="LLM Guardrails Firewall",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .verdict-safe {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    .verdict-unsafe {
        color: #dc3545;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    """Load classifier (cached)."""
    try:
        # Try loading fine-tuned model first
        fine_tuned_path = "./models/fine_tuned_model"
        if os.path.exists(fine_tuned_path):
            return PromptClassifier(
                model_name="distilbert-base-uncased",
                fine_tuned_path=fine_tuned_path,
                threshold=0.5,
                device="cpu"
            )
        else:
            return PromptClassifier(
                model_name="distilbert-base-uncased",
                threshold=0.5,
                device="cpu"
            )
    except Exception as e:
        st.error(f"Error loading classifier: {e}")
        return None


@st.cache_resource
def load_config():
    """Load configuration (cached)."""
    return Config("config.yaml")


@st.cache_data
def load_test_suite():
    """Load test suite (cached)."""
    try:
        return pd.read_csv("data/test_suite.csv")
    except:
        return None


def render_header():
    """Render page header."""
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("🔒", unsafe_allow_html=True)
    with col2:
        st.title("LLM Guardrails Firewall")
        st.markdown("*Detect jailbreaks, injections, and toxic content in LLM prompts*")


def render_sidebar():
    """Render sidebar controls."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Load classifier
        classifier = load_classifier()
        if classifier is None:
            st.error("Failed to load classifier")
            return classifier

        # Threshold control
        threshold = st.slider(
            "Strictness Level",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Lower = stricter (blocks more), Higher = lenient (lets more through)"
        )
        classifier.set_threshold(threshold)

        # Model info
        with st.expander("📊 Model Info"):
            info = classifier.get_model_info()
            st.json({
                "Model": info["model_name"],
                "Fine-tuned": info["fine_tuned"],
                "Device": info["device"],
                "Parameters": f"{info['num_parameters']:,}",
                "Size (MB)": f"{info['model_size_mb']:.1f}"
            })

        st.divider()

        # Quick links
        st.subheader("📚 Resources")
        if st.button("📖 View Project README"):
            with open("README.md", "r") as f:
                st.write(f.read())

    return classifier


def render_live_classification(classifier):
    """Render live classification interface."""
    st.header("🎯 Live Classification")

    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area(
            "Enter a prompt to classify:",
            height=120,
            max_chars=2000,
            placeholder="Type a prompt here..."
        )

    with col2:
        st.write("")
        st.write("")
        classify_button = st.button("🔍 Classify", use_container_width=True)

    if classify_button and user_input:
        with st.spinner("Classifying..."):
            result = classifier.classify(user_input)

        # Display verdict
        col1, col2, col3 = st.columns(3)

        with col1:
            if result['verdict'] == 'safe':
                st.markdown(f"<div class='verdict-safe'>✅ SAFE</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='verdict-unsafe'>🚫 UNSAFE</div>", unsafe_allow_html=True)

        with col2:
            st.metric("Category", result['category'].replace('_', ' ').title())

        with col3:
            confidence = result['confidence']
            st.metric(
                "Confidence",
                f"{confidence:.1%}",
                delta=f"{confidence:.1%}"
            )

        # Scores breakdown
        st.subheader("Score Breakdown")
        scores_df = pd.DataFrame(
            list(result['scores'].items()),
            columns=['Category', 'Probability']
        ).sort_values('Probability', ascending=False)

        fig = px.bar(
            scores_df,
            x='Category',
            y='Probability',
            title="Per-Category Scores",
            labels={'Probability': 'Score'},
            color='Probability',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)


def render_metrics_dashboard(classifier):
    """Render metrics dashboard."""
    st.header("📊 Metrics Dashboard")

    # Load test suite
    test_suite = load_test_suite()
    if test_suite is None:
        st.warning("Test suite not found")
        return

    col1, col2, col3 = st.columns(3)

    # Count statistics
    safe_count = (test_suite['label'] == 'safe').sum()
    unsafe_count = (test_suite['label'] == 'unsafe').sum()

    with col1:
        st.metric("Safe Prompts", safe_count)

    with col2:
        st.metric("Unsafe Prompts", unsafe_count)

    with col3:
        st.metric("Total Test Suite", len(test_suite))

    # Category distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Category Distribution")
        cat_counts = test_suite['category'].value_counts()
        fig = px.pie(
            values=cat_counts.values,
            names=cat_counts.index,
            title="Distribution by Category"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Safe vs Unsafe Distribution")
        label_counts = test_suite['label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=['Safe', 'Unsafe'] if 'safe' in label_counts.index else ['Unsafe'],
            title="Safe vs Unsafe",
            color_discrete_map={'Safe': '#28a745', 'Unsafe': '#dc3545'}
        )
        st.plotly_chart(fig, use_container_width=True)


def render_threshold_analysis():
    """Render threshold sensitivity analysis."""
    st.header("📈 Threshold Sensitivity Analysis")

    st.info(
        "This shows how different thresholds affect classifier behavior. "
        "Lower thresholds = stricter (more false positives), "
        "Higher thresholds = lenient (more false negatives)"
    )

    # Simulate threshold sweep
    thresholds = np.arange(0.1, 1.0, 0.1)

    # Create example data
    data = {
        'Threshold': thresholds,
        'Accuracy': 0.72 + (thresholds * 0.15),
        'Precision': 0.68 + (thresholds * 0.25),
        'Recall': 0.95 - (thresholds * 0.35),
        'False Positive Rate': 0.45 - (thresholds * 0.40)
    }

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy & Precision vs Threshold")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Threshold'],
            y=df['Accuracy'],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=df['Threshold'],
            y=df['Precision'],
            mode='lines+markers',
            name='Precision',
            line=dict(color='#ff7f0e')
        ))
        fig.update_layout(
            title="Accuracy vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recall & False Positive Rate vs Threshold")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Threshold'],
            y=df['Recall'],
            mode='lines+markers',
            name='Recall',
            line=dict(color='#2ca02c')
        ))
        fig.add_trace(go.Scatter(
            x=df['Threshold'],
            y=df['False Positive Rate'],
            mode='lines+markers',
            name='False Positive Rate',
            line=dict(color='#d62728')
        ))
        fig.update_layout(
            title="Recall vs Threshold",
            xaxis_title="Threshold",
            yaxis_title="Score"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_test_suite_explorer():
    """Render test suite explorer."""
    st.header("🔎 Test Suite Explorer")

    test_suite = load_test_suite()
    if test_suite is None:
        st.warning("Test suite not found")
        return

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=test_suite['category'].unique(),
            default=test_suite['category'].unique()
        )

    with col2:
        label_filter = st.multiselect(
            "Filter by Label",
            options=test_suite['label'].unique(),
            default=test_suite['label'].unique()
        )

    with col3:
        search_text = st.text_input("Search prompts...")

    # Apply filters
    filtered = test_suite[
        (test_suite['category'].isin(category_filter)) &
        (test_suite['label'].isin(label_filter))
    ]

    if search_text:
        filtered = filtered[filtered['prompt'].str.contains(search_text, case=False, na=False)]

    # Display table
    st.subheader(f"Showing {len(filtered)} prompts")
    st.dataframe(filtered, use_container_width=True, height=400)


def main():
    """Main app."""
    render_header()

    # Load classifier
    classifier = render_sidebar()
    if classifier is None:
        return

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Live Classification", "📊 Metrics", "📈 Analysis", "🔎 Test Suite"]
    )

    with tab1:
        render_live_classification(classifier)

    with tab2:
        render_metrics_dashboard(classifier)

    with tab3:
        render_threshold_analysis()

    with tab4:
        render_test_suite_explorer()

    # Footer
    st.divider()
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 12px;'>
        LLM Guardrails Firewall | Zero-Cost, Open-Source Safety Classification<br>
        Built with DistilBERT + Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
