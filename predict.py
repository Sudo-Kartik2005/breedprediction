import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_MODEL_PATH = Path("Best_Cattle_Breed.h5")
DEFAULT_CLASS_NAMES = Path("class_names.json")
IMAGE_SIZE = (224, 224)

def load_class_names(class_names_path: Path) -> list:
    with open(class_names_path, "r") as f:
        names = json.load(f)
    return list(names)

def load_image(image_path: Path, image_size=IMAGE_SIZE) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32)
    return arr

def preprocess_for_efficientnet_v2(batch_images: np.ndarray) -> np.ndarray:
    # Match training-time preprocessing
    return tf.keras.applications.efficientnet_v2.preprocess_input(batch_images)

def predict_image(model_path: Path, class_names_path: Path, image_path: Path, top_k: int = 5):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found: {class_names_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    class_names = load_class_names(class_names_path)
    num_classes = len(class_names)
    print(f"Loaded {num_classes} class names")

    image_arr = load_image(image_path, IMAGE_SIZE)
    batch = np.expand_dims(image_arr, axis=0)
    batch = preprocess_for_efficientnet_v2(batch)

    preds = model.predict(batch, verbose=0)[0]
    top_k = max(1, min(top_k, len(preds)))
    top_indices = preds.argsort()[-top_k:][::-1]

    results = [(class_names[i], float(preds[i])) for i in top_indices]
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Predict cattle breed from an image.")
    parser.add_argument("--image", "-i", type=Path, required=True, help="Path to the input image")
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained Keras model (.h5)")
    parser.add_argument("--class-names", "-c", type=Path, default=DEFAULT_CLASS_NAMES, help="Path to class_names.json")
    parser.add_argument("--topk", "-k", type=int, default=5, help="Show top-K predictions")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    results = predict_image(args.model, args.class_names, args.image, top_k=args.topk)
    print("\nPredictions:")
    for rank, (name, prob) in enumerate(results, start=1):
        print(f"{rank}. {name}: {prob*100:.2f}%")


