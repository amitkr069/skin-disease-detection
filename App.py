import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input  # Use DenseNet preprocessing
from PIL import Image

# Load the model
@st.cache_resource
def load_trained_model():
    return load_model("my_model_gpu.keras")

# Load the trained model
model = load_trained_model()

# Class index to label mapping
class_labels = {
    0: "Actinic Keratosis",
    1: "Basal Cell Carcinoma",
    2: "Dermatofibroma",
    3: "Melanoma",
    4: "Nevus",
    5: "Pigmented Benign Keratosis",
    6: "Seborrheic Keratosis",
    7: "Squamous Cell Carcinoma",
    8: "Vascular Lesion"
}

# Streamlit app UI
st.title("Skin Disease Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_shape = (224, 224)  # Correct shape: (width, height)
    img = image.convert("RGB").resize(input_shape)  # Convert to RGB and resize
    img_array = np.array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for DenseNet121

    # Make predictions
    predictions = model.predict(img_array)  # Predict class probabilities
    # st.write(f"Predictions (raw): {predictions}")  # Show raw predictions for debugging

    predicted_class_index = np.argmax(predictions, axis=1)  # Get the class index with the highest probability

    # Get the label from the dictionary
    predicted_class_label = class_labels.get(predicted_class_index[0], "Unknown")

    # Display the result
    st.write(f"Predicted Disease: {predicted_class_label}")
