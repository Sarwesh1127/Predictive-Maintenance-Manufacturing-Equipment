import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🔧 Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_features():
    """Load the trained model and feature names"""
    try:
        model = joblib.load("best_model.pkl")
        feature_names = joblib.load("features_name.pkl")
        # feature_names= {'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'}
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run the Jupyter notebook first to train and save the model.")
        return None, None

def create_input_form(feature_names):
    """Create input form based on actual features"""
    st.subheader("📊 Machine Sensor Readings")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    # Feature descriptions for better UX
    feature_descriptions = {
        'Air temperature [K]': 'Air temperature in Kelvin',
        'Process temperature [K]': 'Process temperature in Kelvin',
        'Rotational speed [rpm]': 'Rotational speed in revolutions per minute',
        'Torque [Nm]': 'Torque in Newton meters',
        'Tool wear [min]': 'Tool wear in minutes',
        'Type': 'Tool Type (Low, Medium, High)',
    }

    input_data = {}

    with col1:
        st.markdown("**Temperature & Speed**")
        for feature in feature_names:
            if 'temperature' in feature.lower() or 'speed' in feature.lower() or 'Torque' in feature:
                if feature in feature_descriptions:
                    help_text = feature_descriptions[feature]
                else:
                    help_text = f"Enter {feature}"

                if 'temperature' in feature.lower():
                    value = st.number_input(f"🌡️ {feature}", min_value=270.0, max_value=400.0, value=300.0, help=help_text)
                elif 'speed' in feature.lower():
                    value = st.number_input(f"⚡ {feature}", min_value=1000, max_value=3000, value=1500, help=help_text)
                elif 'Torque' in feature:
                    value = st.number_input(f"🔧 {feature}", min_value=0.0, max_value=100.0, value=40.0, help=help_text)
                else:
                    value = st.number_input(f"📊 {feature}", value=0.0, help=help_text)
                input_data[feature] = value

    with col2:
        st.markdown("**Tool Wear & Failures**")
        for feature in feature_names:
            if feature not in input_data:
                if feature in feature_descriptions:
                    help_text = feature_descriptions[feature]
                else:
                    help_text = f"Enter {feature}"

                if 'Tool wear' in feature:
                    value = st.number_input(f"🛠️ {feature}", min_value=0, max_value=500, value=100, help=help_text)
                elif 'Type' in feature:
                    value = st.selectbox(f"🧰 {feature}", [0, 1, 2], help=help_text)
                else:
                    value = st.selectbox(f"⚠️ {feature}", [0, 1], help=help_text)
                input_data[feature] = value

    return input_data

def predict_failure(model, input_data, feature_names):
    """Make prediction using the trained model"""
    try:
        # Create input array in correct order
        input_array = np.array([[input_data[feature] for feature in feature_names]])

        # Make prediction
        prediction = model.predict(input_array)
        probabilities = model.predict_proba(input_array)[0]

        return prediction[0], probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

def display_prediction_result(prediction, probabilities):
    """Display prediction results with nice formatting"""
    failure_prob = probabilities[1] * 100

    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-result danger">
                ⚠️ MACHINE FAILURE PREDICTED<br>
                <b>Failure Probability: {failure_prob:.2f}%</b><br>
                <small>Immediate maintenance recommended</small>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-result success">
                ✅ MACHINE OPERATING NORMALLY<br>
                <b>Failure Probability: {failure_prob:.2f}%</b><br>
                <small>Continue regular monitoring</small>
            </div>
        """, unsafe_allow_html=True)

    # Risk level indicator
    if failure_prob > 75:
        risk_level = "🔴 HIGH RISK"
        color = "red"
    elif failure_prob > 50:
        risk_level = "🟠 MEDIUM RISK"
        color = "orange"
    elif failure_prob > 25:
        risk_level = "🟡 LOW RISK"
        color = "yellow"
    else:
        risk_level = "🟢 SAFE"
        color = "green"

    st.markdown(f"**Risk Level:** {risk_level}")

def create_gauge_chart(probability):
    """Create a gauge chart for failure probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Failure Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "lightyellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 75
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<div class="main-header">🔧 Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)

    # Load model and features
    model, feature_names = load_model_and_features()

    if model is None or feature_names is None:
        st.error("⚠️ Model not loaded. Please ensure the model files exist.")
        return

    # Sidebar information
    with st.sidebar:
        st.header("📋 Model Information")
        st.info(f"**Model:** {type(model).__name__}")
        st.info(f"**Features:** {len(feature_names)}")
        # st.info(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        st.header("📖 Instructions")
        st.markdown("""
        1. Enter machine sensor readings
        2. Click 'Predict' to analyze
        3. Review the prediction results
        4. Take appropriate maintenance action
        """)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Analytics", "ℹ️ About"])

    with tab1:
        st.subheader("Machine Health Prediction")

        # Create input form
        input_data = create_input_form(feature_names)

        # Prediction button
        if st.button("🔍 Analyze Machine Health", type="primary", use_container_width=True):
            with st.spinner("Analyzing sensor data..."):
                prediction, probabilities = predict_failure(model, input_data, feature_names)

                if prediction is not None:
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Display prediction result
                        display_prediction_result(prediction, probabilities)

                    with col2:
                        # Show gauge chart
                        st.plotly_chart(create_gauge_chart(probabilities[1]), use_container_width=True)

                    # Additional metrics
                    st.subheader("📈 Detailed Analysis")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Failure Probability", f"{probabilities[1]*100:.1f}%")

                    with col2:
                        st.metric("Normal Probability", f"{probabilities[0]*100:.1f}%")

                    with col3:
                        confidence = max(probabilities) * 100
                        st.metric("Prediction Confidence", f"{confidence:.1f}%")

    with tab2:
        st.subheader("📊 Model Analytics")

        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")

            # Ensure feature_names is a list and matches importances length
            features = list(feature_names)
            importances = model.feature_importances_

            if len(features) == len(importances):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)

                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance Analysis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Mismatch between feature names and importance values.")

        # Model performance metrics
        st.subheader("⚡ Model Performance")
        col1, col2, col3, col4 = st.columns(4)
        metrics = joblib.load("metrics.pkl")

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        with col2:
            st.metric("Precision", f"{metrics['precision']*100:.1f}%")
        with col3:
            st.metric("Recall", f"{metrics['recall']*100:.1f}%")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']*100:.1f}%")

    with tab3:
        st.subheader("ℹ️ About This System")

        st.markdown("""
        **Predictive Maintenance Dashboard** helps you monitor machine health and predict potential failures before they occur.

        **Features:**
        - Real-time machine health prediction
        - Interactive sensor data input
        - Visual analytics and reporting
        - Early warning system for maintenance

        **How it works:**
        1. Enter current sensor readings from your machine
        2. Our trained ML model analyzes the data
        3. Get instant prediction of machine health status
        4. Take preventive maintenance actions as needed

        **Sensor Parameters:**
        - **Temperature readings** in Kelvin
        - **Rotational speed** in RPM
        - **Torque** measurements
        - **Tool wear** indicators
        - **Failure mode** flags
        """)

        st.markdown("---")
        st.markdown("*Built with Streamlit, scikit-learn, and Plotly*")

if __name__ == "__main__":
    main()
