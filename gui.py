import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Load the trained model
model = load_model('/Users/vamsi/venv-metal/interpret.ai/model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('/Users/vamsi/venv-metal/interpret.ai/classes.npy', allow_pickle=True)

# Define image size
IMG_SIZE = 128

def predict_medicine(image):
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    # Make prediction
    predictions = model.predict(image)
    predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
    
    return predicted_label[0]

# Create Gradio interface
iface = gr.Interface(
    fn=predict_medicine, 
    inputs= gr.Image(type="numpy", label="Upload Image"), 
    outputs=gr.Textbox(label="Predicted Medicine Name"),
    title="Interpreter.ai",
    description="Upload an image of handwritten medicine name to interpret it."
)

# Launch the app
iface.launch()
