"""
Similarity Search - Find similar cattle images
Uses feature extraction to find visually similar images
"""

import json
import numpy as np
import tensorflow as tf
from keras_compat import keras
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class SimilaritySearch:
    """Find similar cattle images using deep learning features."""
    
    def __init__(self, model_path: Path, class_names_path: Path, 
                 data_directory: Path = Path("data"), image_size=(224, 224)):
        """
        Initialize similarity search.
        
        Args:
            model_path: Path to trained model
            class_names_path: Path to class names JSON
            data_directory: Directory containing cattle images
            image_size: Target image size
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.data_directory = data_directory
        self.image_size = image_size
        
        # Load model and remove classification head to get features
        self.model = self._load_feature_model()
        self.class_names = self._load_class_names()
        self.image_database = {}
        self.feature_database = None
        
    def _load_class_names(self) -> List[str]:
        """Load class names."""
        with open(self.class_names_path, 'r') as f:
            return json.load(f)
    
    def _load_feature_model(self) -> keras.Model:
        """Load model and create feature extraction model."""
        full_model = keras.models.load_model(self.model_path)
        
        # Get the layer before the final classification layer (usually GlobalAveragePooling2D)
        # This gives us the feature vector
        for i, layer in enumerate(reversed(full_model.layers)):
            if isinstance(layer, keras.layers.GlobalAveragePooling2D):
                # Create model that outputs features
                feature_model = keras.Model(
                    inputs=full_model.input,
                    outputs=layer.output
                )
                return feature_model
        
        # Fallback: use the second-to-last layer
        feature_model = keras.Model(
            inputs=full_model.input,
            outputs=full_model.layers[-2].output
        )
        return feature_model
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for feature extraction."""
        image = image.convert("RGB").resize(self.image_size)
        arr = np.array(image, dtype=np.float32)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
    
    def extract_features(self, image_path: Path) -> np.ndarray:
        """
        Extract feature vector from an image.
        
        Args:
            image_path: Path to image
        
        Returns:
            Feature vector
        """
        img = Image.open(image_path)
        processed = self.preprocess_image(img)
        features = self.model.predict(processed, verbose=0)
        return features[0]  # Remove batch dimension
    
    def build_database(self, max_images_per_breed: int = 50):
        """
        Build feature database from data directory.
        
        Args:
            max_images_per_breed: Maximum images to index per breed
        """
        print("Building similarity search database...")
        
        image_paths = []
        for breed_dir in self.data_directory.iterdir():
            if not breed_dir.is_dir():
                continue
            
            breed_name = breed_dir.name
            images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png")) + list(breed_dir.glob("*.jpeg"))
            images = images[:max_images_per_breed]  # Limit per breed
            
            for img_path in images:
                image_paths.append((img_path, breed_name))
        
        print(f"Found {len(image_paths)} images to index")
        
        # Extract features
        features_list = []
        valid_paths = []
        
        for img_path, breed_name in image_paths:
            try:
                features = self.extract_features(img_path)
                features_list.append(features)
                valid_paths.append((img_path, breed_name))
            except Exception as e:
                print(f"⚠️  Error processing {img_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid images found to index")
        
        self.feature_database = np.array(features_list)
        self.image_database = {
            i: {'path': path, 'breed': breed}
            for i, (path, breed) in enumerate(valid_paths)
        }
        
        print(f"✅ Indexed {len(self.image_database)} images")
    
    def find_similar(self, query_image_path: Path, top_k: int = 5) -> List[Dict]:
        """
        Find similar images to query image.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of similar images to return
        
        Returns:
            List of similar image dictionaries
        """
        if self.feature_database is None:
            raise ValueError("Database not built. Call build_database() first.")
        
        # Extract features from query image
        query_features = self.extract_features(query_image_path)
        query_features = query_features.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_features, self.feature_database)[0]
        
        # Get top-k most similar
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.image_database[idx]['path'],
                'breed': self.image_database[idx]['breed'],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def find_similar_by_breed(self, breed_name: str, top_k: int = 5) -> List[Path]:
        """
        Find images of a specific breed.
        
        Args:
            breed_name: Name of the breed
            top_k: Number of images to return
        
        Returns:
            List of image paths
        """
        breed_images = []
        breed_dir = self.data_directory / breed_name
        
        if not breed_dir.exists():
            return []
        
        images = list(breed_dir.glob("*.jpg")) + list(breed_dir.glob("*.png")) + list(breed_dir.glob("*.jpeg"))
        return images[:top_k]


def similarity_search_cli():
    """Command-line interface for similarity search."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Find similar cattle images")
    parser.add_argument("--query", "-q", type=Path, required=True,
                       help="Query image path")
    parser.add_argument("--model", "-m", type=Path, default=Path("Best_Cattle_Breed.h5"),
                       help="Path to trained model")
    parser.add_argument("--class-names", "-c", type=Path, default=Path("class_names.json"),
                       help="Path to class names JSON")
    parser.add_argument("--data", "-d", type=Path, default=Path("data"),
                       help="Data directory with cattle images")
    parser.add_argument("--topk", "-k", type=int, default=5,
                       help="Number of similar images to return")
    parser.add_argument("--build-db", action="store_true",
                       help="Build database before searching")
    
    args = parser.parse_args()
    
    # Initialize search
    searcher = SimilaritySearch(args.model, args.class_names, args.data)
    
    # Build database if requested
    if args.build_db:
        searcher.build_database()
    else:
        # Try to load existing database or build it
        try:
            searcher.build_database()
        except Exception as e:
            print(f"Error: {e}")
            return
    
    # Find similar images
    print(f"\nSearching for images similar to {args.query}...")
    results = searcher.find_similar(args.query, top_k=args.topk)
    
    print(f"\n{'='*60}")
    print("SIMILAR IMAGES FOUND")
    print(f"{'='*60}\n")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['image_path'].name}")
        print(f"   Breed: {result['breed']}")
        print(f"   Similarity: {result['similarity']*100:.2f}%")
        print(f"   Path: {result['image_path']}\n")


if __name__ == "__main__":
    similarity_search_cli()

