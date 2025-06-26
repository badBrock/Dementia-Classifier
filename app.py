import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Dementia Stage Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #1f2a38 !important;
        color: #f0f4f8 !important;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3a506b;
        margin: 1rem 0;
    }
    .info-box h4 {
        color: #ffcc00 !important;
        margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a67d8 0%, #6b46c1 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


# Load your trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file 'model.pkl' not found. Please ensure the model is trained and saved.")
        return None

model = load_model()

# Feature definitions
feature_definitions = {
    'Visit': 'Visit Number - Sequential visit number for each patient',
    'MR Delay': 'MR Delay - Days between visits',
    'M/F': 'Gender - Male (M) or Female (F) [Use 1 for Male, 0 for Female]',
    'Hand': 'Handedness - Dominant hand [Use 1 for Right, 0 for Left]',
    'Age': 'Age - Patient age in years',
    'EDUC': 'Years of Education - Total years of formal education completed',
    'SES': 'Socioeconomic Status - Socioeconomic status scale (1-5, where 1 is highest)',
    'MMSE': 'Mini Mental State Examination - Cognitive assessment score (0-30)',
    'CDR': 'Clinical Dementia Rating - Dementia severity rating (0, 0.5, 1, 2, 3)',
    'eTIV': 'Estimated Total Intracranial Volume - Brain volume measurement (mm³)',
    'nWBV': 'Normalized Whole Brain Volume - Normalized brain volume ratio',
    'ASF': 'Atlas Scaling Factor - Scaling factor for brain atlas normalization'
}

# Define your selected features
selected_features = [
    'Visit', 'MR Delay', 'M/F', 'Hand', 'Age',
    'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF'
]

# Main title
st.markdown('<h1 class="main-header">🧠 Dementia Stage Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced AI-powered tool for dementia stage classification</p>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("📋 Feature Definitions")
    
    with st.expander("🔍 Click to view all feature descriptions"):
        for feature, description in feature_definitions.items():
            st.markdown(f"**{feature}**: {description}")
    
    st.markdown("---")
    
    st.header("ℹ️ About This Tool")
    st.info("""
    This classifier uses machine learning to predict dementia stages based on clinical and neuroimaging data.
    
    **Classification Categories:**
    - 🟢 **Nondemented**: No signs of dementia
    - 🟡 **Converted**: Mild cognitive impairment progressing to dementia
    - 🔴 **Demented**: Diagnosed with dementia
    """)

# Main content area
if model is not None:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Patient Information Input")
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🧠 Clinical Assessments", "📊 Neuroimaging Data"])
        
        user_input = {}
        
        with tab1:
            st.subheader("Basic Patient Information")
            col_a, col_b = st.columns(2)
            
            with col_a:
                user_input['Visit'] = st.number_input(
                    "Visit Number", 
                    min_value=1, 
                    max_value=10, 
                    value=1,
                    help="Sequential visit number for this patient"
                )
                user_input['Age'] = st.number_input(
                    "Age (years)", 
                    min_value=18, 
                    max_value=120, 
                    value=65,
                    help="Patient's age in years"
                )
                user_input['EDUC'] = st.number_input(
                    "Years of Education", 
                    min_value=0, 
                    max_value=30, 
                    value=12,
                    help="Total years of formal education"
                )
            
            with col_b:
                user_input['MR Delay'] = st.number_input(
                    "MR Delay (days)", 
                    min_value=0, 
                    max_value=365, 
                    value=0,
                    help="Days between visits"
                )
                user_input['M/F'] = st.selectbox(
                    "Gender", 
                    options=[0, 1], 
                    format_func=lambda x: "Female" if x == 0 else "Male",
                    help="Patient's gender"
                )
                user_input['Hand'] = st.selectbox(
                    "Handedness", 
                    options=[0, 1], 
                    format_func=lambda x: "Left" if x == 0 else "Right",
                    help="Patient's dominant hand"
                )
                user_input['SES'] = st.selectbox(
                    "Socioeconomic Status", 
                    options=[1, 2, 3, 4, 5], 
                    index=2,
                    help="Socioeconomic status (1 = highest, 5 = lowest)"
                )
        
        with tab2:
            st.subheader("Clinical Assessment Scores")
            col_c, col_d = st.columns(2)
            
            with col_c:
                user_input['MMSE'] = st.slider(
                    "MMSE Score", 
                    min_value=0, 
                    max_value=30, 
                    value=24,
                    help="Mini Mental State Examination score (0-30, higher is better)"
                )
            
            with col_d:
                user_input['CDR'] = st.selectbox(
                    "CDR Score", 
                    options=[0.0, 0.5, 1.0, 2.0, 3.0], 
                    index=1,
                    help="Clinical Dementia Rating (0 = normal, 3 = severe)"
                )
        
        with tab3:
            st.subheader("Neuroimaging Measurements")
            col_e, col_f = st.columns(2)
            
            with col_e:
                user_input['eTIV'] = st.number_input(
                    "eTIV (mm³)", 
                    min_value=1000.0, 
                    max_value=2500.0, 
                    value=1500.0,
                    help="Estimated Total Intracranial Volume"
                )
                user_input['nWBV'] = st.number_input(
                    "nWBV", 
                    min_value=0.5, 
                    max_value=1.0, 
                    value=0.75, 
                    step=0.01,
                    help="Normalized Whole Brain Volume ratio"
                )
            
            with col_f:
                user_input['ASF'] = st.number_input(
                    "ASF", 
                    min_value=0.5, 
                    max_value=2.0, 
                    value=1.2, 
                    step=0.01,
                    help="Atlas Scaling Factor"
                )
    
    with col2:
        st.header("🔮 Prediction")
        
        if st.button("🚀 Classify Dementia Stage"):
            with st.spinner("Analyzing patient data..."):
                # Convert input to model-ready format
                input_array = np.array([list(user_input.values())])
                prediction = model.predict(input_array)[0]
                
                # Get prediction probabilities if available
                try:
                    probabilities = model.predict_proba(input_array)[0]
                    prob_dict = {
                        "Nondemented": probabilities[0],
                        "Converted": probabilities[1],
                        "Demented": probabilities[2]
                    }
                except:
                    prob_dict = None
                
                label_map = {
                    0: "Nondemented",
                    1: "Converted", 
                    2: "Demented"
                }
                
                result = label_map.get(prediction, 'Unknown')
                
                # Display result with appropriate styling
                if result == "Nondemented":
                    st.success(f"✅ **Prediction: {result}**")
                    st.info("The model indicates no signs of dementia based on the provided data.")
                elif result == "Converted":
                    st.warning(f"⚠️ **Prediction: {result}**")
                    st.info("The model suggests mild cognitive impairment that may progress to dementia.")
                else:
                    st.error(f"🔴 **Prediction: {result}**")
                    st.info("The model indicates signs consistent with dementia.")
                
                # Display probabilities if available
                if prob_dict:
                    st.subheader("📊 Prediction Confidence")
                    for label, prob in prob_dict.items():
                        st.metric(label, f"{prob:.2%}")
        
        # Display current input summary
        st.subheader("📋 Input Summary")
        input_df = pd.DataFrame(list(user_input.items()), columns=['Feature', 'Value'])
        st.dataframe(input_df, use_container_width=True)

    # Footer with disclaimer
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4>⚠️ Medical Disclaimer</h4>
        <p>This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load the model. Please check if 'model.pkl' exists in the current directory.")
    st.info("To use this application, you need to train and save your model first using: `joblib.dump(model, 'model.pkl')`")