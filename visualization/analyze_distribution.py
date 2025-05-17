import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_dataset():
    # Define the dataset directory
    dataset_dir = 'dataset'
    
    # Dictionary to store counts
    class_counts = {}
    
    # Get counts for each class
    for animal_class in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, animal_class)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_counts[animal_class] = len(images)
    
    # Calculate statistics
    counts = list(class_counts.values())
    stats = {
        'Total Images': sum(counts),
        'Average Images per Class': np.mean(counts),
        'Median Images per Class': np.median(counts),
        'Min Images': min(counts),
        'Max Images': max(counts),
        'Std Dev': np.std(counts)
    }
    
    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for stat, value in stats.items():
        print(f"{stat}: {value:.2f}")
    
    # Print class-wise distribution
    print("\nClass-wise Distribution:")
    print("-" * 50)
    for animal_class, count in sorted(class_counts.items()):
        print(f"{animal_class}: {count} images")
    
    # Create a bar plot
    plt.figure(figsize=(15, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Image Distribution Across Animal Classes')
    plt.xlabel('Animal Class')
    plt.ylabel('Number of Images')
    plt.tight_layout()
    plt.savefig('distribution_plot.png')
    plt.close()

if __name__ == "__main__":
    analyze_dataset() 