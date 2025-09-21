# Food Recognition and Nutrition Estimation

This project uses a deep learning model to recognize different types of food from images and estimate their nutritional values (calories, carbs, protein, fat).

## 🚀 Features
- Image classification for multiple food categories (e.g., apple, banana, carrot, etc.)
- Nutrition prediction based on recognized food
- Streamlit web app for interactive use
- Compatible with Google Colab and local environments

## 📂 Project Structure
```
.
├── notebook.ipynb        # Main Jupyter Notebook (model training & testing)
├── streamlit_predict.py  # Streamlit app for food prediction
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore file
```

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/food-recognition.git
   cd food-recognition
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv food-venv
   source food-venv/bin/activate   # On Windows: food-venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
### Run Jupyter Notebook
```bash
jupyter notebook
```

### Run Streamlit App
```bash
streamlit run streamlit_predict.py
```

Upload an image of food and get its predicted nutrition values.

## 📊 Dataset
- Supports datasets like **Fruits and Vegetables Image Dataset**
- Preprocessing & augmentation done in the notebook

## 🛠️ Tech Stack
- Python 3.x
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Streamlit
- PIL (Python Imaging Library)

## ✨ Future Improvements
- Expand dataset with more food categories
- Improve accuracy with transfer learning (MobileNetV2, EfficientNet, etc.)
- Add allergen detection and dietary suggestions

---
👩‍💻 Developed by [Your Name]
