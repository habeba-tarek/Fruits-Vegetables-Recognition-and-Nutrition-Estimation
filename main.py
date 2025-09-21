# streamlit_nutrition.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Fruit & Vegetable Nutrition Predictor", page_icon="🥗")
st.title("🥦 Fruit & Vegetable Nutrition Predictor")
st.write("Upload an image and get the predicted fruit/vegetable with its nutrition info and possible allergens!")

# Load the trained model
@st.cache_resource
def load_fv_model():
    model = load_model('FV.h5')
    return model

model = load_fv_model()

# Class labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell_pepper', 4: 'cabbage',
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chili_pepper', 9: 'corn',
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes',
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango',
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'peach', 24: 'pear',
    25: 'peas', 26: 'pineapple', 27: 'pomegranate', 28: 'potato', 29: 'raddish',
    30: 'soybean', 31: 'spinach', 32: 'sweet_potato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

# Nutrition info dictionary (per 100g)
nutrition_info = {
    'apple':      {'Calories': 52,  'Carbs(g)': 14, 'Protein(g)': 0.3, 'Fat(g)': 0.2},
    'banana':     {'Calories': 96,  'Carbs(g)': 27, 'Protein(g)': 1.3, 'Fat(g)': 0.3},
    'beetroot':   {'Calories': 43,  'Carbs(g)': 10, 'Protein(g)': 1.6, 'Fat(g)': 0.2},
    'bell_pepper':{'Calories': 31,  'Carbs(g)': 6,  'Protein(g)': 1,   'Fat(g)': 0.3},
    'cabbage':    {'Calories': 25,  'Carbs(g)': 6,  'Protein(g)': 1.3, 'Fat(g)': 0.1},
    'capsicum':   {'Calories': 20,  'Carbs(g)': 4.6,'Protein(g)': 0.9, 'Fat(g)': 0.2},
    'carrot':     {'Calories': 41,  'Carbs(g)': 10, 'Protein(g)': 0.9, 'Fat(g)': 0.2},
    'cauliflower':{'Calories': 25,  'Carbs(g)': 5,  'Protein(g)': 2,   'Fat(g)': 0.3},
    'chili_pepper':{'Calories': 40, 'Carbs(g)': 9,  'Protein(g)': 2,   'Fat(g)': 0.4},
    'corn':       {'Calories': 86,  'Carbs(g)': 19, 'Protein(g)': 3.2, 'Fat(g)': 1.2},
    'cucumber':   {'Calories': 16,  'Carbs(g)': 3.6,'Protein(g)': 0.7, 'Fat(g)': 0.1},
    'eggplant':   {'Calories': 25,  'Carbs(g)': 6,  'Protein(g)': 1,   'Fat(g)': 0.2},
    'garlic':     {'Calories': 149, 'Carbs(g)': 33, 'Protein(g)': 6.4, 'Fat(g)': 0.5},
    'ginger':     {'Calories': 80,  'Carbs(g)': 18, 'Protein(g)': 1.8, 'Fat(g)': 0.8},
    'grapes':     {'Calories': 69,  'Carbs(g)': 18, 'Protein(g)': 0.7, 'Fat(g)': 0.2},
    'jalepeno':   {'Calories': 29,  'Carbs(g)': 6,  'Protein(g)': 1,   'Fat(g)': 0.4},
    'kiwi':       {'Calories': 61,  'Carbs(g)': 15, 'Protein(g)': 1.1, 'Fat(g)': 0.5},
    'lemon':      {'Calories': 29,  'Carbs(g)': 9,  'Protein(g)': 1.1, 'Fat(g)': 0.3},
    'lettuce':    {'Calories': 15,  'Carbs(g)': 2.9,'Protein(g)': 1.4, 'Fat(g)': 0.2},
    'mango':      {'Calories': 60,  'Carbs(g)': 15, 'Protein(g)': 0.8, 'Fat(g)': 0.4},
    'onion':      {'Calories': 40,  'Carbs(g)': 9.3,'Protein(g)': 1.1, 'Fat(g)': 0.1},
    'orange':     {'Calories': 47,  'Carbs(g)': 12, 'Protein(g)': 0.9, 'Fat(g)': 0.1},
    'paprika':    {'Calories': 282, 'Carbs(g)': 54, 'Protein(g)': 14,  'Fat(g)': 13},
    'peach':      {'Calories': 39,  'Carbs(g)': 10, 'Protein(g)': 0.9, 'Fat(g)': 0.3},
    'pear':       {'Calories': 57,  'Carbs(g)': 15, 'Protein(g)': 0.4, 'Fat(g)': 0.1},
    'peas':       {'Calories': 81,  'Carbs(g)': 14, 'Protein(g)': 5.4, 'Fat(g)': 0.4},
    'pineapple':  {'Calories': 50,  'Carbs(g)': 13, 'Protein(g)': 0.5, 'Fat(g)': 0.1},
    'pomegranate':{'Calories': 83,  'Carbs(g)': 19, 'Protein(g)': 1.7, 'Fat(g)': 1.2},
    'potato':     {'Calories': 77,  'Carbs(g)': 17, 'Protein(g)': 2,   'Fat(g)': 0.1},
    'raddish':    {'Calories': 16,  'Carbs(g)': 3.4,'Protein(g)': 0.7, 'Fat(g)': 0.1},
    'soybean':    {'Calories': 173, 'Carbs(g)': 9,  'Protein(g)': 16.6,'Fat(g)': 9},
    'spinach':    {'Calories': 23,  'Carbs(g)': 3.6,'Protein(g)': 2.9, 'Fat(g)': 0.4},
    'sweet_potato':{'Calories': 86, 'Carbs(g)': 20, 'Protein(g)': 1.6, 'Fat(g)': 0.1},
    'tomato':     {'Calories': 18,  'Carbs(g)': 3.9,'Protein(g)': 0.9, 'Fat(g)': 0.2},
    'turnip':     {'Calories': 28,  'Carbs(g)': 6.4,'Protein(g)': 0.9, 'Fat(g)': 0.1},
    'watermelon': {'Calories': 30,  'Carbs(g)': 8,  'Protein(g)': 0.6, 'Fat(g)': 0.2},
}

# Allergens / Sensitivity info dictionary
allergen_info = {
    'apple': ['Pollen (Birch)', 'Oral Allergy Syndrome'],
    'banana': ['Latex-fruit syndrome'],
    'beetroot': [],
    'bell_pepper': ['Nightshade sensitivity'],
    'cabbage': [],
    'capsicum': ['Nightshade sensitivity'],
    'carrot': ['Pollen (Birch)', 'Oral Allergy Syndrome'],
    'cauliflower': [],
    'chili_pepper': ['Nightshade sensitivity'],
    'corn': [],
    'cucumber': [],
    'eggplant': ['Nightshade sensitivity'],
    'garlic': [],
    'ginger': [],
    'grapes': ['Grape allergy'],
    'jalepeno': ['Nightshade sensitivity'],
    'kiwi': ['Actinidin allergy'],
    'lemon': ['Citrus allergy'],
    'lettuce': [],
    'mango': ['Urushiol (similar to poison ivy)'],
    'onion': [],
    'orange': ['Citrus allergy'],
    'paprika': ['Nightshade sensitivity'],
    'peach': ['Stone fruit allergy'],
    'pear': [],
    'peas': ['Legume allergy'],
    'pineapple': ['Bromelain sensitivity'],
    'pomegranate': [],
    'potato': ['Nightshade sensitivity'],
    'raddish': [],
    'soybean': ['Soy allergy'],
    'spinach': [],
    'sweet_potato': [],
    'tomato': ['Nightshade sensitivity'],
    'turnip': [],
    'watermelon': [],
}

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    def preprocess_image(image_file):
        img = load_img(image_file, target_size=(224,224))
        img = img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        return img

    img_array = preprocess_image(uploaded_file)

    # Predict
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[predicted_class_idx]
    
    st.success(f"Predicted: **{predicted_label}**")

    # Show nutrition info if available
    if predicted_label in nutrition_info:
        st.subheader("Nutrition Information (per 100g)")
        info = nutrition_info[predicted_label]
        for key, value in info.items():
            st.write(f"{key}: {value}")
    else:
        st.write("Nutrition information not available for this item.")

    # Show allergens if available
    if predicted_label in allergen_info and allergen_info[predicted_label]:
        st.subheader("⚠️ Possible Allergens / Sensitivities")
        for allergen in allergen_info[predicted_label]:
            st.write(f"- {allergen}")
    else:
        st.write("No common allergens reported for this item.")
