import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import joblib

def load_model(model_path, scaler_path):
    """Load the trained SVM model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def setup_feature_extractor():
    """Set up ResNet50 model for feature extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, device

def preprocess_image(image_path, transform, feature_extractor, device):
    """Preprocess a single image for model prediction."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Extract features using ResNet50
        with torch.no_grad():
            features = feature_extractor(image)
        
        # Process features
        features = features.squeeze().cpu().numpy()
        features = features.reshape(1, -1)
        
        return features
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def test_images(model, scaler, test_dir, class_names, feature_extractor, transform, device):
    """Test the model on all images in the test directory."""
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
    
    # Process each animal class
    for animal in os.listdir(test_dir):
        animal_dir = os.path.join(test_dir, animal)
        if not os.path.isdir(animal_dir):
            continue
            
        print(f"\nTesting images for {animal}...")
        
        # Process each image in the animal's directory
        image_files = [f for f in os.listdir(animal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for img_name in tqdm(image_files):
            img_path = os.path.join(animal_dir, img_name)
            
            try:
                # Preprocess and predict
                features = preprocess_image(img_path, transform, feature_extractor, device)
                if features is None:
                    continue
                    
                # Scale features
                scaled_features = scaler.transform(features)
                
                # Predict
                predicted_class_idx = model.predict(scaled_features)[0]
                predicted_class = class_names[predicted_class_idx]
                
                # Update results
                results[animal]['total'] += 1
                if predicted_class == animal:
                    results[animal]['correct'] += 1
                
                # Store prediction details
                results[animal]['predictions'].append({
                    'image': img_name,
                    'predicted': predicted_class,
                    'correct': predicted_class == animal
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue
    
    return results

def save_results(results, output_file='test_results.json'):
    """Save test results to a JSON file."""
    # Calculate overall statistics
    total_correct = sum(data['correct'] for data in results.values())
    total_images = sum(data['total'] for data in results.values())
    
    # Prepare results with accuracy metrics
    formatted_results = {
        'overall_accuracy': total_correct / total_images if total_images > 0 else 0,
        'class_results': {}
    }
    
    for animal, data in results.items():
        accuracy = data['correct'] / data['total'] if data['total'] > 0 else 0
        formatted_results['class_results'][animal] = {
            'accuracy': accuracy,
            'correct': data['correct'],
            'total': data['total'],
            'predictions': data['predictions']
        }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(formatted_results, f, indent=4)
    
    return formatted_results

def main():
    # Configuration
    model_path = 'pro - 1/models/svm_model.joblib'
    scaler_path = 'pro - 1/models/scaler.joblib'
    test_dir = 'pro - 1/dataset/test'
    
    # Load class names from the model's classes
    class_mapping = {
        'Bear': 0, 'Bird': 1, 'Cat': 2, 'Cow': 3, 'Deer': 4, 'Dog': 5, 'Dolphin': 6,
        'Elephant': 7, 'Giraffe': 8, 'Horse': 9, 'Kangaroo': 10, 'Lion': 11,
        'Panda': 12, 'Tiger': 13, 'Zebra': 14
    }
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
    
    # Set up feature extractor and transforms
    feature_extractor, device = setup_feature_extractor()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model and scaler
    print("Loading model and scaler...")
    model, scaler = load_model(model_path, scaler_path)
    
    # Test images
    print("Testing images...")
    results = test_images(model, scaler, test_dir, class_names, feature_extractor, transform, device)
    
    # Save and display results
    formatted_results = save_results(results)
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Overall Accuracy: {formatted_results['overall_accuracy']:.2%}")
    print("\nPer-class Results:")
    for animal, data in formatted_results['class_results'].items():
        print(f"{animal}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")

if __name__ == "__main__":
    main() 