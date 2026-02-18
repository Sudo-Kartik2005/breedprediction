"""
Explainable AI - Grad-CAM Visualization
Shows which parts of the image the model focuses on for predictions
"""

import numpy as np
import tensorflow as tf
from keras_compat import keras
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to use. If None, uses last conv layer.
        """
        self.model = model
        self.layer_name = layer_name
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
                    self.layer_name = layer.name
                    break
        
        if self.layer_name is None:
            raise ValueError("No convolutional layer found in the model")
        
        # Create a model that outputs the feature maps and predictions
        self.grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            img_array: Preprocessed image array
            pred_index: Class index to generate heatmap for. If None, uses predicted class.
        
        Returns:
            Heatmap array
        """
        # Get predictions
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image.
        
        Args:
            img: Original PIL Image
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay
        
        Returns:
            PIL Image with heatmap overlay
        """
        # Resize heatmap to match image size
        img_array = np.array(img)
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        
        # Overlay
        overlayed = alpha * heatmap_colored + (1 - alpha) * img_array
        overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(overlayed)
    
    def visualize(self, img, img_array, class_names, top_k=1):
        """
        Generate complete visualization with original image, heatmap, and overlay.
        
        Args:
            img: Original PIL Image
            img_array: Preprocessed image array
            class_names: List of class names
            top_k: Number of top predictions to visualize
        
        Returns:
            Dictionary with visualization components
        """
        # Get predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-top_k:][::-1]
        
        results = {}
        
        for idx in top_indices:
            breed_name = class_names[idx]
            confidence = predictions[idx]
            
            # Generate heatmap
            heatmap = self.make_gradcam_heatmap(img_array, pred_index=idx)
            
            # Create overlay
            overlay = self.overlay_heatmap(img, heatmap)
            
            results[breed_name] = {
                'confidence': float(confidence),
                'heatmap': heatmap,
                'overlay': overlay
            }
        
        return results


def preprocess_image_for_gradcam(image_path: Path, target_size=(224, 224)):
    """Preprocess image for Grad-CAM visualization."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def visualize_gradcam(model_path: Path, image_path: Path, class_names_path: Path, 
                      output_path: Path = None, top_k: int = 1):
    """
    Complete Grad-CAM visualization pipeline.
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        class_names_path: Path to class names JSON
        output_path: Optional path to save visualization
        top_k: Number of top predictions to visualize
    """
    # Load model and class names
    model = keras.models.load_model(model_path)
    with open(class_names_path, 'r') as f:
        import json
        class_names = json.load(f)
    
    # Preprocess image
    img, img_array = preprocess_image_for_gradcam(image_path)
    
    # Create Grad-CAM
    gradcam = GradCAM(model)
    
    # Generate visualizations
    results = gradcam.visualize(img, img_array, class_names, top_k=top_k)
    
    # Display results
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Visualizations for each top prediction
    for i, (breed_name, data) in enumerate(results.items(), 1):
        if i < len(axes):
            axes[i].imshow(data['overlay'])
            axes[i].set_title(f"{breed_name}\nConfidence: {data['confidence']*100:.2f}%")
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for cattle breed classification")
    parser.add_argument("--image", "-i", type=Path, required=True, help="Path to input image")
    parser.add_argument("--model", "-m", type=Path, default=Path("Best_Cattle_Breed.h5"), 
                       help="Path to trained model")
    parser.add_argument("--class-names", "-c", type=Path, default=Path("class_names.json"),
                       help="Path to class names JSON")
    parser.add_argument("--output", "-o", type=Path, help="Path to save visualization")
    parser.add_argument("--topk", "-k", type=int, default=1, help="Number of top predictions to visualize")
    
    args = parser.parse_args()
    visualize_gradcam(args.model, args.image, args.class_names, args.output, args.topk)

