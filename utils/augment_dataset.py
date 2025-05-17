import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image
import random

def load_image(image_path):
    """Load an image and convert to RGB."""
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def save_image(image, save_path):
    """Save the augmented image."""
    try:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
        return True
    except Exception as e:
        print(f"Error saving image {save_path}: {str(e)}")
        return False

def create_augmentation_pipeline():
    """Create an augmentation pipeline with various transformations."""
    return A.Compose([
        # Geometric Transformations
        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        ], p=0.7),
        
        # Flips
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.3),
        ], p=0.5),
        
        # Color Transformations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ColorJitter(p=0.5),
        ], p=0.7),
        
        # Noise and Blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=0.4),
        
        # Weather and Lighting
        A.OneOf([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.5),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.5),
        ], p=0.3),
        
        # Quality and Sharpness
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
        ], p=0.5),
        
        # Crop and Padding
        A.OneOf([
            A.RandomCrop(height=224, width=224, p=0.3),
            A.CenterCrop(height=224, width=224, p=0.3),
            A.Resize(height=224, width=224, p=0.4),
        ], p=0.5),
    ])

def augment_class(class_path, target_count):
    """Augment images in a class directory until reaching target count."""
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    current_count = len(images)
    
    if current_count >= target_count:
        return
    
    augmentation = create_augmentation_pipeline()
    needed_count = target_count - current_count
    
    pbar = tqdm(total=needed_count, desc=f"Augmenting {os.path.basename(class_path)}")
    
    while current_count < target_count:
        # Randomly select an image to augment
        source_image_name = random.choice(images)
        source_image_path = os.path.join(class_path, source_image_name)
        
        # Load and augment image
        image = load_image(source_image_path)
        if image is None:
            continue
        
        # Apply multiple augmentations to the same image
        num_augmentations = min(3, target_count - current_count)
        for _ in range(num_augmentations):
            # Apply augmentation
            augmented = augmentation(image=image)
            augmented_image = augmented['image']
            
            # Save augmented image
            base_name = os.path.splitext(source_image_name)[0]
            aug_name = f"{base_name}_aug_{current_count}.jpg"
            save_path = os.path.join(class_path, aug_name)
            
            if save_image(augmented_image, save_path):
                current_count += 1
                pbar.update(1)
            
            if current_count >= target_count:
                break
    
    pbar.close()

def balance_dataset(target_count=450):
    """Balance the entire dataset through augmentation."""
    dataset_dir = 'dataset'
    print(f"Balancing all classes to {target_count} images")
    
    # Augment each class to match the target count
    for animal_class in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, animal_class)
        if os.path.isdir(class_path):
            augment_class(class_path, target_count)

if __name__ == "__main__":
    # Install required package if not already installed
    try:
        import albumentations
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "albumentations"])
    
    balance_dataset(450)  # Set target to 450 images per class
    
    # Run distribution analysis after augmentation
    from analyze_distribution import analyze_dataset
    analyze_dataset() 