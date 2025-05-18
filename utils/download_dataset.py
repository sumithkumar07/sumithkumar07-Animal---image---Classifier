import os
import requests
import logging
from tqdm import tqdm
import shutil
import zipfile
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Animal classes to download
ANIMAL_CLASSES = [
    'cat', 'dog', 'horse', 'elephant', 'butterfly',
    'chicken', 'cow', 'sheep', 'spider', 'squirrel',
    'tiger', 'wolf', 'zebra', 'deer', 'duck'
]

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'dataset/raw',
        'dataset/processed',
        'dataset/augmented',
        'models',
        'uploads',
        'test_images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_sample_images():
    """
    This is a placeholder function. In a real implementation, you would:
    1. Use an image dataset API (like Flickr, Google Images, or Kaggle)
    2. Download images for each animal class
    3. Save them to dataset/raw/{class_name}
    
    For now, we'll create empty directories and instruct users.
    """
    raw_dir = 'dataset/raw'
    
    # Create directories for each class
    for class_name in ANIMAL_CLASSES:
        class_dir = os.path.join(raw_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        logger.info(f"Created directory for {class_name}: {class_dir}")
    
    # Print instructions for users
    print("\nDataset Directory Structure Created!")
    print("=====================================")
    print("\nTo prepare your dataset:")
    print("1. Collect images for each animal class (minimum 30 images per class recommended)")
    print("2. Place the images in their respective directories under dataset/raw/")
    print("   Example: dataset/raw/cat/cat1.jpg")
    print("\nImage Requirements:")
    print("- Supported formats: .jpg, .jpeg, .png, .webp")
    print("- Minimum size: 224x224 pixels")
    print("- Maximum size: 4096x4096 pixels")
    print("\nNext Steps:")
    print("1. Run data_preprocessing.py to process the raw images")
    print("2. Run augment_dataset.py to augment the dataset")
    print("3. Run feature_extractor.py to extract features")
    print("4. Run train_svm.py to train the model")
    print("\nNote: You can use tools like the Flickr API, Google Images, or Kaggle")
    print("to download images for your dataset.")

def main():
    """Main function to set up the project structure."""
    try:
        logger.info("Starting project setup...")
        
        # Create directory structure
        setup_directories()
        
        # Create dataset structure
        download_sample_images()
        
        logger.info("Project setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in project setup: {str(e)}")
        raise

if __name__ == "__main__":
    main() 