import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from huggingface_hub import hf_hub_download
import logging
import os
import base64
from io import BytesIO
from datetime import datetime
import uuid
import tempfile
import google.generativeai as genai
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Chest X-Ray Pneumonia Detector",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google API Key - Streamlit Secrets se lega
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")

# Model Configuration - Aapka Hugging Face token yahan use karein
HF_TOKEN = st.secrets.get("HF_TOKEN", "")  # Yeh aapne jo token liya hai
REPO_ID = "DanishMubashar/chest-x-ray-pneumonia-detection-DenseNet121"
FILENAME = "pneumonia_detection_model.keras"
IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.5

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0066cc, #003366);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .diagnosis-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .normal-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .pneumonia-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff9900;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #0066cc;
    }
    .report-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== GEMINI SETUP ====================
class AIGenerator:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = None
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini initialized")
            except Exception as e:
                logger.error(f"❌ Gemini init failed: {e}")
    
    def generate_text(self, prompt):
        """Generate text using Gemini or return None"""
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}")
        return None

# Initialize AI Generator
ai_gen = AIGenerator(GOOGLE_API_KEY)

# ==================== MODEL LOADING ====================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load pneumonia detection model with caching"""
    try:
        with st.spinner("📥 Downloading model from HuggingFace... Please wait"):
            logger.info("📥 Downloading model from HuggingFace...")
            
            # Download model with token
            model_path = hf_hub_download(
                repo_id=REPO_ID, 
                filename=FILENAME,
                token=HF_TOKEN if HF_TOKEN else None
            )
            logger.info(f"✅ Model downloaded: {model_path}")
            
            # Load model
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                # Test with dummy input
                _ = model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
                logger.info("✅ Model loaded successfully")
                return model
            except Exception as e:
                logger.warning(f"⚠️ Standard load failed, trying rebuild: {e}")
                return rebuild_model(model_path)
                
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        st.error(f"Model loading failed: {e}")
        st.warning("⚠️ Using fallback model for testing")
        return create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model"""
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def rebuild_model(model_path):
    """Rebuild model architecture and load weights"""
    base_model = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        pooling='max',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.load_weights(model_path)
    logger.info("✅ Model rebuilt successfully")
    return model

# ==================== IMAGE PROCESSING ====================
def preprocess_image(img):
    """Preprocess image for model input"""
    if img is None:
        return None
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    # Add batch dimension
    img_input = np.expand_dims(img_array, axis=0)
    
    return img_input

def generate_gradcam(img_array, original_img, model):
    """Generate Grad-CAM heatmap"""
    try:
        if img_array is None or original_img is None:
            return original_img
            
        # Get base model
        base_model = model.layers[0] if isinstance(model, tf.keras.Sequential) else model
        
        # Find last conv layer
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if not last_conv_layer:
            return original_img
        
        # Create grad model
        grad_model = tf.keras.models.Model(
            [base_model.inputs],
            [base_model.get_layer(last_conv_layer).output, base_model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Pass through remaining layers if Sequential
            if isinstance(model, tf.keras.Sequential):
                x = predictions
                for layer in model.layers[1:]:
                    x = layer(x)
                class_channel = x[:, 0]
            else:
                class_channel = predictions[:, 0]
        
        # Calculate gradients
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Generate heatmap
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        orig_np = np.array(original_img)
        heatmap = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert to RGB if grayscale
        if len(orig_np.shape) == 2:
            orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
        
        # Overlay
        superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(superimposed)
        
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}")
        return original_img

# ==================== AI FUNCTIONS ====================
def get_ai_insights(diagnosis, confidence, age, gender):
    """Get AI-generated clinical insights"""
    
    # Try Gemini first
    if ai_gen.model:
        prompt = f"""As an AI medical assistant, provide brief clinical insights for a chest X-ray.
Patient: {age} years old, {gender}
Diagnosis: {diagnosis} with {confidence*100:.1f}% confidence
Provide 2-3 sentences of clinical interpretation. Be professional and concise."""
        
        response = ai_gen.generate_text(prompt)
        if response:
            return response
    
    # Fallback responses
    if diagnosis == "PNEUMONIA":
        return f"The chest X-ray shows findings consistent with pneumonia ({confidence*100:.1f}% confidence). There are visible opacities in the lung fields suggesting infection. Clinical correlation with symptoms like fever, cough, and difficulty breathing is recommended."
    else:
        return f"The chest X-ray appears normal with clear lung fields ({confidence*100:.1f}% confidence). No visible infiltrates, consolidations, or abnormalities detected. If symptoms persist, further clinical evaluation may be needed."

def get_recommendations(diagnosis, age, gender):
    """Get AI-generated recommendations"""
    
    # Try Gemini first
    if ai_gen.model:
        prompt = f"""Based on chest X-ray diagnosis: {diagnosis}
Patient age: {age}, gender: {gender}
Provide brief medical recommendations (2-3 sentences). Include immediate steps and follow-up advice."""
        
        response = ai_gen.generate_text(prompt)
        if response:
            return response
    
    # Fallback responses
    try:
        age_num = int(age)
    except:
        age_num = 30
        
    if diagnosis == "PNEUMONIA":
        if age_num < 18:
            return "👶 Consult pediatrician immediately. Monitor for breathing difficulty, ensure adequate hydration, and complete full course of prescribed antibiotics. Follow-up in 2 weeks is recommended."
        elif age_num > 65:
            return "👴 Seek immediate medical attention. Elderly patients with pneumonia require careful monitoring. Complete prescribed antibiotics, rest, and stay hydrated. Follow-up chest X-ray may be needed."
        else:
            return "👨 Consult physician for treatment plan. Rest, hydration, and prescribed medications are essential. Seek emergency care if breathing difficulty worsens. Follow-up in 2-3 weeks."
    else:
        return "✅ No immediate medical action needed based on X-ray. Continue routine health practices. If symptoms like cough, fever, or breathing difficulty persist, consult healthcare provider for clinical correlation."

# ==================== REPORT FUNCTIONS ====================
def generate_html_report(data):
    """Generate HTML report"""
    html = f"""
    <div class="report-card">
        <h2 style="color: #0066cc; text-align: center;">🏥 Chest X-Ray Analysis Report</h2>
        <p style="text-align: center; color: #666;">Generated: {data['generation_date']}</p>
        
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h3>📋 Patient Information</h3>
            <p><strong>Name:</strong> {data['patient_name']}</p>
            <p><strong>Age:</strong> {data['patient_age']} years</p>
            <p><strong>Gender:</strong> {data['patient_gender']}</p>
            <p><strong>Report ID:</strong> {data['report_id']}</p>
        </div>
        
        <div style="background: {'#f8d7da' if data['diagnosis'] == 'PNEUMONIA' else '#d4edda'}; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h2>{'🔴' if data['diagnosis'] == 'PNEUMONIA' else '🟢'} Diagnosis: {data['diagnosis']}</h2>
            <p><strong>Confidence:</strong> {data['confidence']*100:.1f}%</p>
            <div style="background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="background: #0066cc; width: {data['confidence']*100}%; height: 100%;"></div>
            </div>
        </div>
        
        <div class="info-box">
            <h3>🤖 AI Clinical Insights</h3>
            <p>{data['ai_insights']}</p>
        </div>
        
        <div class="recommendation-box">
            <h3>📋 Medical Recommendations</h3>
            <p>{data['recommendations']}</p>
        </div>
        
        <p style="font-size: 0.8rem; color: #999; margin-top: 2rem;">
            <strong>Disclaimer:</strong> This is an AI-generated report. Always consult a healthcare professional.
        </p>
    </div>
    """
    return html

# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown('<div class="main-header"><h1>🏥 Chest X-Ray Pneumonia Detection System</h1><p>AI-Powered Diagnostic Assistant with Gemini Integration</p></div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model... Please wait"):
        model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face token.")
        st.stop()
    
    # Sidebar for patient info
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital.png", width=100)
        st.markdown("## 📝 Patient Information")
        
        patient_name = st.text_input("Full Name", placeholder="Enter patient name")
        
        col1, col2 = st.columns(2)
        with col1:
            patient_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        with col2:
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        st.markdown("---")
        st.markdown("## 📤 Upload X-Ray")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png", "dicom"],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
        
        st.markdown("---")
        analyze_btn = st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True)
        
        # Show API status
        if GOOGLE_API_KEY:
            st.success("✅ Gemini AI Connected")
        else:
            st.warning("⚠️ Gemini API Key not set")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Analysis Results")
        result_placeholder = st.empty()
        
    with col2:
        st.markdown("### 🔬 Grad-CAM Visualization")
        heatmap_placeholder = st.empty()
    
    # Report section
    st.markdown("### 📋 Detailed Report")
    report_placeholder = st.empty()
    
    # Status bar
    status_placeholder = st.info("👆 Fill patient info and upload X-ray to begin analysis")
    
    # Analysis logic
    if analyze_btn and uploaded_file is not None and patient_name:
        with st.spinner("🔄 Analyzing X-ray... Please wait"):
            try:
                # Preprocess image
                img_input = preprocess_image(image)
                
                # Make prediction
                prediction = model.predict(img_input, verbose=0)[0][0]
                
                # Determine diagnosis
                is_pneumonia = prediction > CONFIDENCE_THRESHOLD
                confidence = float(prediction) if is_pneumonia else 1 - float(prediction)
                diagnosis = "PNEUMONIA" if is_pneumonia else "NORMAL"
                
                # Generate heatmap
                if is_pneumonia:
                    result_image = generate_gradcam(img_input, image, model)
                else:
                    result_image = image
                
                # Generate AI insights
                insights = get_ai_insights(diagnosis, confidence, patient_age, patient_gender)
                recommendations = get_recommendations(diagnosis, patient_age, patient_gender)
                
                # Display results
                with col1:
                    result_placeholder.image(image, caption="Original X-Ray", use_container_width=True)
                
                with col2:
                    heatmap_placeholder.image(result_image, caption="Analysis Result", use_container_width=True)
                
                # Display diagnosis box
                if diagnosis == "PNEUMONIA":
                    st.markdown(f"""
                    <div class="diagnosis-box pneumonia-box">
                        <h2>🔴 Diagnosis: PNEUMONIA DETECTED</h2>
                        <p>Confidence: {confidence*100:.1f}%</p>
                        <div style="background: #e0e0e0; height: 25px; border-radius: 12px; overflow: hidden;">
                            <div style="background: #dc3545; width: {confidence*100}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="diagnosis-box normal-box">
                        <h2>🟢 Diagnosis: NORMAL</h2>
                        <p>Confidence: {confidence*100:.1f}%</p>
                        <div style="background: #e0e0e0; height: 25px; border-radius: 12px; overflow: hidden;">
                            <div style="background: #28a745; width: {confidence*100}%; height: 100%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display insights and recommendations
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>🤖 AI Clinical Insights</h4>
                        <p>{insights}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h4>📋 Medical Recommendations</h4>
                        <p>{recommendations}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate report ID
                report_id = f"XRAY-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
                
                # Prepare report data
                report_data = {
                    "patient_name": patient_name,
                    "patient_age": patient_age,
                    "patient_gender": patient_gender,
                    "report_id": report_id,
                    "generation_date": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "ai_insights": insights,
                    "recommendations": recommendations
                }
                
                # Generate and display HTML report
                html_report = generate_html_report(report_data)
                report_placeholder.markdown(html_report, unsafe_allow_html=True)
                
                # Status update
                status_placeholder.success(f"✅ Analysis complete! Diagnosis: {diagnosis} ({confidence*100:.1f}% confidence)")
                
            except Exception as e:
                status_placeholder.error(f"❌ Error: {str(e)}")
    
    elif analyze_btn and not patient_name:
        status_placeholder.warning("⚠️ Please enter patient name")
    elif analyze_btn and uploaded_file is None:
        status_placeholder.warning("⚠️ Please upload an X-ray image")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>⚠️ <strong>Important Disclaimer:</strong> This is an AI-powered tool for educational purposes only. 
        All diagnoses should be confirmed by qualified healthcare professionals.</p>
        <p>© 2024 Chest X-Ray Pneumonia Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
