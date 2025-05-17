import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def count_original_and_augmented(class_path):
    """Count original and augmented images in a class directory."""
    original = 0
    augmented = 0
    for filename in os.listdir(class_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            if '_aug_' in filename:
                augmented += 1
            else:
                original += 1
    return original, augmented

def visualize_distribution():
    dataset_dir = 'dataset'
    classes = []
    original_counts = []
    augmented_counts = []
    
    # Collect data
    for animal_class in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, animal_class)
        if os.path.isdir(class_path):
            orig, aug = count_original_and_augmented(class_path)
            classes.append(animal_class)
            original_counts.append(orig)
            augmented_counts.append(aug)
    
    # Create figure with subplots
    plt.figure(figsize=(20, 10))
    
    # 1. Bar plot showing original and augmented images
    plt.subplot(1, 2, 1)
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, original_counts, width, label='Original Images', color='blue', alpha=0.7)
    plt.bar(x + width/2, augmented_counts, width, label='Augmented Images', color='green', alpha=0.7)
    
    plt.xlabel('Animal Classes')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Original vs Augmented Images per Class')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Pie chart showing total distribution
    plt.subplot(1, 2, 2)
    total_images = [orig + aug for orig, aug in zip(original_counts, augmented_counts)]
    plt.pie(total_images, labels=classes, autopct='%1.1f%%', startangle=90)
    plt.title('Proportion of Images per Class')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nDetailed Dataset Statistics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Original':<10} {'Augmented':<10} {'Total':<10} {'% Augmented':<10}")
    print("-" * 60)
    
    total_orig = sum(original_counts)
    total_aug = sum(augmented_counts)
    
    for i, class_name in enumerate(classes):
        orig = original_counts[i]
        aug = augmented_counts[i]
        total = orig + aug
        aug_percent = (aug / total) * 100
        print(f"{class_name:<15} {orig:<10} {aug:<10} {total:<10} {aug_percent:>6.1f}%")
    
    print("-" * 60)
    print(f"{'Total':<15} {total_orig:<10} {total_aug:<10} {total_orig + total_aug:<10}")
    print(f"\nTotal Dataset Size: {total_orig + total_aug} images")
    print(f"Average images per class: {(total_orig + total_aug) / len(classes):.1f}")
    print(f"Percentage of augmented images: {(total_aug / (total_orig + total_aug)) * 100:.1f}%")

if __name__ == "__main__":
    visualize_distribution() 