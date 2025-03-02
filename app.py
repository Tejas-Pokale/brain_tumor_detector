import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained Keras model
# MODEL_PATH = "models/1.keras"
# model = tf.keras.models.load_model(MODEL_PATH)

# Load the TFLite model
MODEL_PATH = "models/3.tflite"
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(MODEL_PATH)

def predict(image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])

    return predictions

# Class labels
CLASS_NAMES = ["Glioma", "Meningioma","No Tumor", "Pituitary"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to match model input
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0) 
    return image

# Streamlit Page Config
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# Global Styling
st.markdown("""
    <style>
        body, .stApp {
            background-color: black !important;
            color: white !important;
        }
        h1, h2, h3 {
            color: red !important;
            text-align: center;
        }
        .stFileUploader label {
            color: white !important;
            font-size: 18px !important;
            font-weight: bold !important;
        }
        .stButton>button {
            background-color: red !important;
            color: white !important;
            border-radius: 10px;
            font-size: 16px;
            padding: 8px 20px;
        }
        .stImage img {
            border: 3px solid red !important;
            border-radius: 10px;
            box-shadow: 0px 0px 10px red;
        }
        .tumor-text {
            color: red;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
        }
        .no-tumor-text {
            color: white;
            font-size: 22px;
            font-weight: bold;
            text-align: center;
        }
        .confidence-table {
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }
        .confidence-table th, .confidence-table td {
            border: 1px solid red;
            padding: 10px;
            text-align: center;
            color: white;
        }
        .confidence-table th {
            background-color: red;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🧠 Brain Tumor Detection</h1>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("📂 Upload a Brain MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("<h3>📸 Uploaded Image</h3>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)

    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        #prediction = model.predict(processed_image)   # if using keras or H5 model
        prediction = predict(processed_image)    # if using tflite model
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence_scores = [round(score * 100, 2) for score in prediction[0]]

        # Display result with proper styling
        if predicted_class == "No Tumor":
            st.markdown(f"<p class='no-tumor-text'>✅ {predicted_class} detected (Confidence: {confidence_scores[np.argmax(prediction)]:.2f}%)</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='tumor-text'>⚠️ {predicted_class} detected! (Confidence: {confidence_scores[np.argmax(prediction)]:.2f}%)</p>", unsafe_allow_html=True)

        # Show confidence scores in table format
        st.markdown("<h3>🔬 Confidence Scores for All Classes</h3>", unsafe_allow_html=True)
        table_html = "<table class='confidence-table'><tr><th>Class</th><th>Confidence</th></tr>"
        for i, class_name in enumerate(CLASS_NAMES):
            table_html += f"<tr><td>{class_name}</td><td>{confidence_scores[i]}%</td></tr>"
        table_html += "</table>"
    
        st.markdown(table_html, unsafe_allow_html=True)
    except:
        st.markdown("<h1>Unable to predict for the given image ,\nmake sure to upload right image</h1>", unsafe_allow_html=True)
