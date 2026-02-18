"""
Batch Processing for Multiple Images
Process multiple cattle images at once and generate reports
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import tensorflow as tf
from keras_compat import keras
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class BatchProcessor:
    """Process multiple images in batch for cattle breed classification."""
    
    def __init__(self, model_path: Path, class_names_path: Path, image_size=(224, 224)):
        """
        Initialize batch processor.
        
        Args:
            model_path: Path to trained model
            class_names_path: Path to class names JSON
            image_size: Target image size for preprocessing
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.image_size = image_size
        self.model = None
        self.class_names = None
        self._load_resources()
    
    def _load_resources(self):
        """Load model and class names."""
        print(f"Loading model from {self.model_path}...")
        self.model = tf.keras.models.load_model(self.model_path)
        
        with open(self.class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"✅ Loaded model with {len(self.class_names)} classes")
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess a single image."""
        image = image.convert("RGB").resize(self.image_size)
        arr = np.array(image, dtype=np.float32)
        arr = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
        return arr
    
    def predict_batch(self, image_paths: List[Path], top_k: int = 5, 
                     batch_size: int = 32) -> List[Dict]:
        """
        Process multiple images and return predictions.
        
        Args:
            image_paths: List of image file paths
            top_k: Number of top predictions to return
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load and preprocess images
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path)
                    processed = self.preprocess_image(img)
                    batch_images.append(processed)
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    continue
            
            if not batch_images:
                continue
            
            # Batch prediction
            batch_array = np.array(batch_images)
            predictions = self.model.predict(batch_array, verbose=0)
            
            # Process results
            for j, pred in enumerate(predictions):
                top_indices = pred.argsort()[-top_k:][::-1]
                top_predictions = [
                    {
                        'breed': self.class_names[idx],
                        'confidence': float(pred[idx])
                    }
                    for idx in top_indices
                ]
                
                results.append({
                    'image_path': str(valid_paths[j]),
                    'image_name': valid_paths[j].name,
                    'predictions': top_predictions,
                    'top_breed': top_predictions[0]['breed'],
                    'top_confidence': top_predictions[0]['confidence']
                })
        
        return results
    
    def process_directory(self, directory: Path, top_k: int = 5, 
                         extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'),
                         recursive: bool = True) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            directory: Directory containing images
            top_k: Number of top predictions
            extensions: Image file extensions to process
            recursive: Whether to search subdirectories
        
        Returns:
            List of prediction dictionaries
        """
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        # Find all image files
        image_paths = []
        if recursive:
            for ext in extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
        
        if not image_paths:
            print(f"⚠️  No images found in {directory}")
            return []
        
        print(f"Found {len(image_paths)} images to process")
        return self.predict_batch(image_paths, top_k=top_k)
    
    def generate_report(self, results: List[Dict], output_path: Path = None, 
                       format: str = 'csv') -> pd.DataFrame:
        """
        Generate a summary report from batch results.
        
        Args:
            results: List of prediction results
            output_path: Optional path to save report
            format: Output format ('csv', 'excel', 'json')
        
        Returns:
            DataFrame with results
        """
        # Flatten results for DataFrame
        rows = []
        for result in results:
            row = {
                'image_name': result['image_name'],
                'image_path': result['image_path'],
                'predicted_breed': result['top_breed'],
                'confidence': result['top_confidence'],
                'confidence_percent': result['top_confidence'] * 100
            }
            
            # Add top-k predictions
            for i, pred in enumerate(result['predictions'], 1):
                row[f'breed_{i}'] = pred['breed']
                row[f'confidence_{i}'] = pred['confidence'] * 100
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Save if output path provided
        if output_path:
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'excel':
                df.to_excel(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2)
            print(f"✅ Report saved to {output_path}")
        
        return df
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Generate statistics from batch results.
        
        Args:
            results: List of prediction results
        
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}
        
        # Breed distribution
        breed_counts = {}
        confidences = []
        
        for result in results:
            breed = result['top_breed']
            breed_counts[breed] = breed_counts.get(breed, 0) + 1
            confidences.append(result['top_confidence'])
        
        stats = {
            'total_images': len(results),
            'unique_breeds': len(breed_counts),
            'average_confidence': np.mean(confidences) * 100,
            'min_confidence': np.min(confidences) * 100,
            'max_confidence': np.max(confidences) * 100,
            'breed_distribution': dict(sorted(breed_counts.items(), key=lambda x: x[1], reverse=True))
        }
        
        return stats


def process_batch_cli():
    """Command-line interface for batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process cattle breed images")
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="Input directory or file containing images")
    parser.add_argument("--model", "-m", type=Path, default=Path("Best_Cattle_Breed.h5"),
                       help="Path to trained model")
    parser.add_argument("--class-names", "-c", type=Path, default=Path("class_names.json"),
                       help="Path to class names JSON")
    parser.add_argument("--output", "-o", type=Path, help="Output path for report")
    parser.add_argument("--format", "-f", choices=['csv', 'excel', 'json'], default='csv',
                       help="Output format")
    parser.add_argument("--topk", "-k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search subdirectories")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchProcessor(args.model, args.class_names)
    
    # Process images
    if args.input.is_file():
        results = processor.predict_batch([args.input], top_k=args.topk, batch_size=args.batch_size)
    else:
        results = processor.process_directory(args.input, top_k=args.topk, recursive=args.recursive)
    
    if not results:
        print("No results to process.")
        return
    
    # Generate report
    if args.output:
        processor.generate_report(results, args.output, format=args.format)
    else:
        df = processor.generate_report(results)
        print("\n" + "="*50)
        print("BATCH PROCESSING RESULTS")
        print("="*50)
        print(df.head(10).to_string())
        print(f"\n... and {len(df) - 10} more rows")
    
    # Print statistics
    stats = processor.get_statistics(results)
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Unique breeds detected: {stats['unique_breeds']}")
    print(f"Average confidence: {stats['average_confidence']:.2f}%")
    print(f"Confidence range: {stats['min_confidence']:.2f}% - {stats['max_confidence']:.2f}%")
    print("\nTop 10 breeds detected:")
    for breed, count in list(stats['breed_distribution'].items())[:10]:
        print(f"  {breed}: {count} images")


if __name__ == "__main__":
    process_batch_cli()

