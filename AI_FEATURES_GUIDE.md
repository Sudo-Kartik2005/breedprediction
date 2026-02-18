# 🤖 AI Features Guide

This document describes all the AI-powered features added to the Cattle Breed Classification system.

## 📋 Table of Contents

1. [AI Chatbot](#ai-chatbot)
2. [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
3. [Batch Processing](#batch-processing)
4. [Similarity Search](#similarity-search)
5. [Enhanced Web App](#enhanced-web-app)
6. [Usage Examples](#usage-examples)

---

## 1. AI Chatbot 💬

**File:** `ai_chatbot.py`

An intelligent chatbot that answers questions about cattle breeds using a knowledge base.

### Features:
- Natural language understanding for breed-related queries
- Information about breed origins, characteristics, and types
- List all available breeds
- Answer questions about dairy vs draft breeds
- Conversational interface

### Usage:

**Command Line:**
```bash
python ai_chatbot.py
```

**In Code:**
```python
from ai_chatbot import CattleBreedChatbot

chatbot = CattleBreedChatbot()
response = chatbot.respond("Tell me about Gir cattle")
print(response)
```

### Example Queries:
- "Tell me about Gir cattle"
- "What dairy breeds do you know?"
- "List all breeds"
- "What is the origin of Sahiwal?"

---

## 2. Explainable AI (Grad-CAM) 👁️

**File:** `explainable_ai.py`

Visualizes which parts of an image the model focuses on when making predictions using Gradient-weighted Class Activation Mapping.

### Features:
- Heatmap visualization showing model attention
- Overlay on original image
- Support for top-K predictions
- Helps understand model decision-making

### Usage:

**Command Line:**
```bash
python explainable_ai.py --image path/to/image.jpg --output visualization.png
```

**In Code:**
```python
from explainable_ai import GradCAM, visualize_gradcam
from pathlib import Path

model = tf.keras.models.load_model("Best_Cattle_Breed.h5")
gradcam = GradCAM(model)

# Generate heatmap
heatmap = gradcam.make_gradcam_heatmap(img_array)
overlay = gradcam.overlay_heatmap(original_image, heatmap)
```

### What it shows:
- **Red/Yellow areas**: Where the model focuses most
- **Blue areas**: Less important regions
- Helps verify the model is looking at relevant features (cattle body, not background)

---

## 3. Batch Processing 📦

**File:** `batch_processor.py`

Process multiple images at once for efficient classification of large datasets.

### Features:
- Process entire directories or multiple files
- Batch prediction for speed
- Generate CSV/Excel/JSON reports
- Statistics and breed distribution analysis
- Progress tracking with tqdm

### Usage:

**Command Line:**
```bash
# Process a directory
python batch_processor.py --input data/ --output results.csv --format csv

# Process specific files
python batch_processor.py --input image1.jpg image2.jpg --output results.csv
```

**In Code:**
```python
from batch_processor import BatchProcessor
from pathlib import Path

processor = BatchProcessor(
    Path("Best_Cattle_Breed.h5"),
    Path("class_names.json")
)

# Process directory
results = processor.process_directory(Path("test_images/"))

# Generate report
df = processor.generate_report(results, Path("report.csv"), format="csv")

# Get statistics
stats = processor.get_statistics(results)
print(f"Average confidence: {stats['average_confidence']:.2f}%")
```

### Output Format:
- CSV: Spreadsheet-compatible
- Excel: .xlsx format with formatting
- JSON: Structured data format

---

## 4. Similarity Search 🔎

**File:** `similarity_search.py`

Find visually similar cattle images using deep learning feature extraction.

### Features:
- Build searchable database from image directory
- Find similar images using cosine similarity
- Feature extraction from trained model
- Fast similarity matching

### Usage:

**Command Line:**
```bash
# Build database and search
python similarity_search.py --query query_image.jpg --build-db --topk 5
```

**In Code:**
```python
from similarity_search import SimilaritySearch
from pathlib import Path

searcher = SimilaritySearch(
    Path("Best_Cattle_Breed.h5"),
    Path("class_names.json"),
    Path("data")
)

# Build database
searcher.build_database(max_images_per_breed=50)

# Find similar images
results = searcher.find_similar(Path("query.jpg"), top_k=5)

for result in results:
    print(f"{result['image_path']}: {result['similarity']*100:.2f}% similar")
```

### Use Cases:
- Find similar cattle for breeding selection
- Quality control - find outliers
- Dataset exploration
- Find images of the same breed

---

## 5. Enhanced Web App 🌐

**File:** `app_enhanced.py`

A comprehensive Streamlit web application integrating all AI features.

### Features:
- **Tab 1: Classification** - Original image classification
- **Tab 2: AI Chatbot** - Interactive breed information chatbot
- **Tab 3: Explainable AI** - Grad-CAM visualization
- **Tab 4: Batch Processing** - Process multiple images
- **Tab 5: Similarity Search** - Find similar images

### Running the App:

```bash
streamlit run app_enhanced.py
```

### Features in the App:

1. **Unified Interface**: All features accessible from one app
2. **Real-time Processing**: Instant results with progress indicators
3. **Visual Feedback**: Charts, heatmaps, and image displays
4. **Export Options**: Download results as CSV
5. **Responsive Design**: Works on different screen sizes

---

## 6. Usage Examples

### Example 1: Complete Workflow

```python
# 1. Classify an image
from predict import predict_image
results = predict_image(
    Path("Best_Cattle_Breed.h5"),
    Path("class_names.json"),
    Path("cattle.jpg"),
    top_k=5
)

# 2. Ask chatbot about the result
from ai_chatbot import CattleBreedChatbot
chatbot = CattleBreedChatbot()
info = chatbot.respond(f"Tell me about {results[0][0]}")

# 3. Visualize what the model sees
from explainable_ai import visualize_gradcam
visualize_gradcam(
    Path("Best_Cattle_Breed.h5"),
    Path("cattle.jpg"),
    Path("class_names.json"),
    Path("visualization.png")
)

# 4. Find similar images
from similarity_search import SimilaritySearch
searcher = SimilaritySearch(...)
searcher.build_database()
similar = searcher.find_similar(Path("cattle.jpg"), top_k=5)
```

### Example 2: Batch Analysis

```python
from batch_processor import BatchProcessor

processor = BatchProcessor(...)
results = processor.process_directory(Path("farm_photos/"))

# Analyze results
stats = processor.get_statistics(results)
print(f"Most common breed: {max(stats['breed_distribution'], key=stats['breed_distribution'].get)}")
print(f"Average confidence: {stats['average_confidence']:.2f}%")

# Export
processor.generate_report(results, Path("farm_analysis.csv"))
```

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the enhanced app:**
   ```bash
   streamlit run app_enhanced.py
   ```

3. **Or use individual modules:**
   ```bash
   # Chatbot
   python ai_chatbot.py
   
   # Batch processing
   python batch_processor.py --input data/ --output results.csv
   
   # Similarity search
   python similarity_search.py --query image.jpg --build-db
   ```

---

## 📊 Technical Details

### AI Models Used:
- **Classification**: EfficientNetV2 (pre-trained, fine-tuned)
- **Feature Extraction**: Same model, feature layer extraction
- **Similarity**: Cosine similarity on feature vectors

### Performance:
- **Batch Processing**: ~32 images/second (depends on hardware)
- **Similarity Search**: Database building: ~1-2 sec/image, Search: <100ms
- **Grad-CAM**: ~1-2 seconds per image

### Dependencies:
- TensorFlow 2.18.0
- scikit-learn (for similarity metrics)
- OpenCV (for image processing)
- Streamlit (for web interface)
- Pandas (for data handling)

---

## 🎯 Future Enhancements

Potential additions:
- [ ] Real-time video classification
- [ ] Mobile app integration
- [ ] Cloud deployment
- [ ] Model fine-tuning interface
- [ ] Advanced analytics dashboard
- [ ] Multi-model ensemble
- [ ] Active learning for model improvement

---

## 📝 Notes

- All AI features require the trained model (`Best_Cattle_Breed.h5`)
- Similarity search requires the `data/` directory with breed images
- Grad-CAM works best with clear, well-lit images
- Batch processing is optimized for GPU but works on CPU

---

## 🤝 Contributing

Feel free to extend these features:
- Add more breed information to the chatbot
- Improve Grad-CAM visualizations
- Add more export formats
- Enhance similarity search algorithms

---

**Happy Classifying! 🐄🤖**

