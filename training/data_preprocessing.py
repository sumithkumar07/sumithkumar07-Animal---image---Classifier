import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm
import shutil
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_images_per_class(dataset_path):
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

def create_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, p=0.3),
            A.GridDistortion(distort_limit=0.1, p=0.1),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ], p=0.3),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.Resize(224, 224),  # Add consistent resizing
    ])

def augment_image(image, augmentation):
    augmented = augmentation(image=image)
    return augmented['image']

def balance_dataset(dataset_path, target_samples=500):
    """Balance the dataset by augmenting underrepresented classes"""
    print("Balancing dataset...")
    
    # Create augmented directory if it doesn't exist
    augmented_path = os.path.join(os.path.dirname(dataset_path), "dataset_augmented")
    if os.path.exists(augmented_path):
        shutil.rmtree(augmented_path)
    os.makedirs(augmented_path)
    
    # Get current class counts
    class_counts = count_images_per_class(dataset_path)
    augmentation = create_augmentation_pipeline()
    
    for class_name, count in class_counts.items():
        print(f"\nProcessing class: {class_name}")
        source_dir = os.path.join(dataset_path, class_name)
        target_dir = os.path.join(augmented_path, class_name)
        os.makedirs(target_dir)
        
        # Copy original images
        images = os.listdir(source_dir)
        for img_name in images:
            shutil.copy2(
                os.path.join(source_dir, img_name),
                os.path.join(target_dir, img_name)
            )
        
        # Augment if needed
        if count < target_samples:
            num_augmentations = target_samples - count
            print(f"Generating {num_augmentations} augmented images...")
            
            # Read all original images
            original_images = []
            for img_name in images:
                img_path = os.path.join(source_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:  # Check if image was loaded successfully
                    original_images.append(img)
            
            if not original_images:
                print(f"Warning: No valid images found in {class_name}")
                continue
                
            for i in tqdm(range(num_augmentations)):
                # Randomly select an image to augment
                img = original_images[np.random.randint(0, len(original_images))]
                augmented_img = augment_image(img, augmentation)
                
                # Save augmented image
                cv2.imwrite(
                    os.path.join(target_dir, f'aug_{i}.jpg'),
                    augmented_img
                )

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'dataset/raw',
        'dataset/processed',
        'dataset/augmented'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def process_images(min_size=224):
    """Process raw images and save them to the processed directory."""
    raw_dir = 'dataset/raw'
    processed_dir = 'dataset/processed'
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get list of class directories
    class_dirs = [d for d in os.listdir(raw_dir) 
                 if os.path.isdir(os.path.join(raw_dir, d))]
    
    total_processed = 0
    skipped = 0
    
    for class_name in class_dirs:
        # Create class directory in processed
        processed_class_dir = os.path.join(processed_dir, class_name)
        os.makedirs(processed_class_dir, exist_ok=True)
        
        # Get all images in the class directory
        raw_class_dir = os.path.join(raw_dir, class_name)
        images = [f for f in os.listdir(raw_class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        logger.info(f"Processing {len(images)} images for class: {class_name}")
        
        for image_name in tqdm(images, desc=f"Processing {class_name}"):
            try:
                # Open and verify image
                image_path = os.path.join(raw_class_dir, image_name)
                img = Image.open(image_path)
                
                # Convert RGBA to RGB if necessary
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode not in ['RGB']:
                    img = img.convert('RGB')
                
                # Check image dimensions
                if min(img.size) < min_size:
                    logger.warning(f"Image too small, skipping: {image_path}")
                    skipped += 1
                    continue
                
                # Save processed image
                processed_path = os.path.join(processed_class_dir, image_name)
                img.save(processed_path, format='JPEG', quality=95)
                total_processed += 1
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {str(e)}")
                skipped += 1
                continue
    
    logger.info(f"Processing complete. Processed: {total_processed}, Skipped: {skipped}")
    return total_processed, skipped

def create_class_mapping():
    """Create a mapping between class names and indices."""
    processed_dir = 'dataset/processed'
    
    # Get sorted list of class names
    class_names = sorted([d for d in os.listdir(processed_dir) 
                         if os.path.isdir(os.path.join(processed_dir, d))])
    
    # Create mapping
    class_mapping = {name: idx for idx, name in enumerate(class_names)}
    
    # Save mapping
    np.save('models/class_mapping.npy', class_mapping)
    logger.info(f"Created class mapping with {len(class_names)} classes")
    
    return class_mapping

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        logger.info("Starting preprocessing pipeline...")
        
        # Setup directories
        setup_directories()
        
        # Process images
        total_processed, skipped = process_images()
        
        # Create class mapping
        class_mapping = create_class_mapping()
        
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info(f"Total images processed: {total_processed}")
        logger.info(f"Total images skipped: {skipped}")
        logger.info(f"Number of classes: {len(class_mapping)}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 