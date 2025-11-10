import os
import json
import numpy as np
import cv2
import tensorflow as tf
import warnings
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
warnings.filterwarnings("ignore")

# ------------ CONSTANTS -----------------
MODEL_PATH ="Best_Cattle_Breed.h5"   #your trained cattle breed model
DATA_DIR = "data/"
CLASS_NAMES_FILE = "class_names.json"

# -------------------- LOAD MODEL --------------------------
model = tf.keras.models.load_model(MODEL_PATH)
# Get number of classes from model output shape
NUM_CLASSES = model.output_shape[1]

# Try to load class names from saved file first
CLASS_NAMES = None
if os.path.exists(CLASS_NAMES_FILE):
    try:
        with open(CLASS_NAMES_FILE, 'r') as f:
            CLASS_NAMES = json.load(f)
        if len(CLASS_NAMES) == NUM_CLASSES:
            print(f"✅ Loaded {len(CLASS_NAMES)} class names from {CLASS_NAMES_FILE}")
        else:
            print(f"Warning: {CLASS_NAMES_FILE} has {len(CLASS_NAMES)} classes but model expects {NUM_CLASSES}.")
            CLASS_NAMES = None
    except Exception as e:
        print(f"Error loading {CLASS_NAMES_FILE}: {e}")
        CLASS_NAMES = None

# If file doesn't exist, try to get from data directory
if CLASS_NAMES is None:
    if os.path.exists(DATA_DIR) and os.path.isdir(DATA_DIR):
        CLASS_NAMES = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
        if len(CLASS_NAMES) == NUM_CLASSES:
            print(f"✅ Loaded {len(CLASS_NAMES)} classes from data directory")
            # Save for future use
            with open(CLASS_NAMES_FILE, 'w') as f:
                json.dump(CLASS_NAMES, f)
        else:
            print(f"Warning: Found {len(CLASS_NAMES)} directories but model expects {NUM_CLASSES} classes.")
            CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]
    else:
        # If data directory doesn't exist, create placeholder class names
        print(f"⚠️  Warning: Data directory not found and {CLASS_NAMES_FILE} not found.")
        print(f"   Using placeholder class names (Class_0 to Class_{NUM_CLASSES-1}).")
        print(f"   To get actual breed names, either:")
        print(f"   1. Download the dataset and create the data/ directory structure")
        print(f"   2. Or run train.py to generate {CLASS_NAMES_FILE}")
        CLASS_NAMES = [f"Class_{i}" for i in range(NUM_CLASSES)]

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or invalid path.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    return img

# ---------------- PREDICTION FUNCTION ------------------------
def predict_image(image_path):
    try:
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        preds = model.predict(img, verbose=0)
        class_id = np.argmax(preds[0])
        confidence = preds[0][class_id] * 100
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None
    
# ------------------ GUI CALLBACKS ----------------------------
def browse_image():
    file_path = filedialog.askopenfilename(
        title=" Select Cattle Image",
        filetypes=[(" Image files", "*.jpg *.jpeg *.png")]
    ) 
    if file_path:
        image = Image.open(file_path)
        image = image.resize((300, 300))
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo

        predicted_class, confidence = predict_image(file_path)
        if predicted_class:
            result_label.config(
                text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
            )


# --------------------- GUI SETUP -----------------------
root = Tk()
root.title("Cattle Breed Classifier")
root.geometry("400x500")

Label(root, text="Indian Cattle Breed Classifier", font=("Arial", 16)).pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

browse_btn = Button(root, text= "Select Cattle Image", command=browse_image)
browse_btn.pack(pady=20)

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

root.mainloop()
