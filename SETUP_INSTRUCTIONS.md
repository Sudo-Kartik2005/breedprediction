# Setup Instructions for Cattle Breed Classification

## Setting Up the Dataset

1. **Download the dataset from Kaggle:**
   - Link: https://www.kaggle.com/datasets/lukex9442/indian-bovine-breeds
   - Download the dataset (usually a ZIP file)

2. **Extract the dataset:**
   - Extract the ZIP file
   - The extracted folder should contain subdirectories for each breed
   - Each subdirectory should contain images of that breed

3. **Create the data directory:**
   - Rename the extracted folder to `data` (if it's not already named `data`)
   - OR copy the extracted folder to this project directory and rename it to `data`
   - The structure should look like:
     ```
     data/
     ├── Alambadi/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     ├── Amritmahal/
     │   ├── image1.jpg
     │   └── ...
     ├── Ayrshire/
     └── ... (other breed folders)
     ```

4. **Extract class names:**
   - Run: `python extract_class_names.py`
   - This will create `class_names.json` with all breed names

5. **Run the application:**
   - Run: `python chatbot.py`
   - Now predictions will show actual breed names!

## Quick Setup Commands

If your dataset is already extracted somewhere:
1. Copy the dataset folder to this directory
2. Rename it to `data` (if needed)
3. Run: `python extract_class_names.py`
4. Run: `python chatbot.py`

