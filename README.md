# 🐄🔍 Cattle Breed Classification System 🚀 | Computer Vision + Deep Learning Project

<img width="1280" height="720" alt="SQL_Thumbnail (32)" src="https://github.com/user-attachments/assets/9d2dd8ee-7f67-44c3-a450-9bbcd232d785" />

🚀 Learn how I built a Cattle Breed Classifier using Python, TensorFlow & Deep Learning. Identify breeds like Alambadi, Amritmahal, Ayrshire, Banni, and more directly from cattle images with AI + GUI. 🐄🤖

💡 Ever wondered if Artificial Intelligence can help farmers, researchers, and veterinarians identify cattle breeds faster?
In this video, I’ll show you how I built a Cattle Breed Classifier using Python, TensorFlow, and Deep Learning — a complete end-to-end AI project for agriculture & livestock management. 🐮📊

We’ll go step by step:
✅ Loading and preprocessing cattle images
✅ Training a Deep Learning model with EfficientNetV2B0
✅ Building a classifier for multiple cattle breeds 🐄
✅ Creating a GUI with Tkinter for easy image upload & prediction
✅ Displaying results with breed name & confidence percentage 🎯

This is a full End-to-End Machine Learning Project — perfect for students, beginners, or anyone who wants to add an AI + Agriculture + Computer Vision project to their portfolio. 🌱

✨ By the end, you’ll learn how to:
• Prepare and organize a cattle image dataset
• Train & test a Deep Learning breed classifier
• Build a user-friendly GUI for predictions
• Apply AI in agriculture & livestock research

📌 Technologies Used: Python, TensorFlow, Keras, OpenCV, Tkinter, NumPy

💬 If you’re starting in AI/ML, this tutorial shows how to turn an idea into a working real-world agricultural AI application.

🔔 Subscribe for more AI, ML, and Python projects: @SouvikChai

📢 Share this project with friends who love AI in Agriculture & Computer Vision! 🌾

## ⚡ Quickstart

1) Create a Python environment (recommended Python 3.10 or 3.11), then install dependencies:

```bash
pip install -r requirements.txt
```

2) Prepare the dataset into `data/` and generate `class_names.json`:

```bash
python setup_from_archive.py --archive "C:\Users\asus\Downloads\archive" --target data
```

Use `--dry-run` to preview actions:

```bash
python setup_from_archive.py --dry-run
```

3) (Optional) Train the model:

```bash
python train.py
```

This will save the best model to `Best_Cattle_Breed.h5` and class names to `class_names.json`.

4) Run prediction on an image (CLI):

```bash
python predict.py --image path/to/image.jpg
```

Show top-3 predictions with confidences:

```bash
python predict.py -i path/to/image.jpg -k 3
```

Advanced options:

```bash
python predict.py --image img.jpg --model Best_Cattle_Breed.h5 --class-names class_names.json --topk 5
```

## 🌐 Streamlit Web App (Improved Frontend)

Launch a simple web UI for uploading images and viewing top‑K predictions:

```bash
streamlit run app.py
```

Sidebar lets you pick the model and class names files. Upload a `.jpg/.jpeg/.png` to see predictions and a confidence bar chart.

## 📂 Project Structure

- `setup_from_archive.py`: Discover/copy dataset from an archive into `data/` and generate `class_names.json` (now with CLI arguments and `--dry-run`).
- `train.py`: Train EfficientNetV2B0 classifier on images in `data/` (224×224).
- `predict.py`: Simple CLI inference tool to get top‑K breed predictions for a single image.
- `app.py`: Streamlit web app for an improved frontend experience.
- `Best_Cattle_Breed.h5`: Trained model checkpoint (created by training).
- `class_names.json`: Class label list corresponding to training order.
- `data/`: Image dataset laid out as one folder per class.
- `requirements.txt`: Pinned dependencies for reproducible setup.

## 🛠 Notes

- Input size expected by the model is 224×224 RGB and uses EfficientNetV2 preprocessing, which `predict.py` applies automatically.
- If you already have `data/`, you can regenerate `class_names.json` by running the setup script again pointing `--target` to your data folder.