import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image

def create_augmentation_pipeline():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomBrightness(limit=0.2, p=1),
            A.RandomContrast(limit=0.2, p=1),
            A.RandomGamma(p=1)
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.ISONoise(p=1),
            A.MultiplicativeNoise(p=1)
        ], p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1)
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.ElasticTransform(p=1)
        ], p=0.3),
        A.Resize(224, 224)
    ])

def augment_images(base_dir, num_augmented_per_image=3):
    """
    Augment images for challenging classes
    """
    # Classes that need more augmentation based on model performance
    challenging_classes = ['Dog', 'Horse', 'Cat', 'Elephant', 'Zebra']
    transform = create_augmentation_pipeline()
    
    for animal in challenging_classes:
        print(f"\nAugmenting images for {animal}...")
        animal_dir = os.path.join(base_dir, animal)
        
        if not os.path.exists(animal_dir):
            print(f"Directory not found for {animal}")
            continue
            
        # Get list of existing images
        image_files = [f for f in os.listdir(animal_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"Processing {animal} images"):
            try:
                # Read image
                img_path = os.path.join(animal_dir, img_file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate augmented versions
                for i in range(num_augmented_per_image):
                    augmented = transform(image=image)
                    aug_image = augmented['image']
                    
                    # Convert to PIL Image and save
                    aug_image = Image.fromarray(aug_image)
                    base_name = os.path.splitext(img_file)[0]
                    aug_path = os.path.join(animal_dir, 
                                          f"{base_name}_aug_{i+1}.jpg")
                    aug_image.save(aug_path, quality=95)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue

def main():
    base_dir = 'pro - 1/dataset/test'
    print("Starting image augmentation process...")
    augment_images(base_dir, num_augmented_per_image=5)
    print("\nAugmentation completed!")

if __name__ == "__main__":
    main() 