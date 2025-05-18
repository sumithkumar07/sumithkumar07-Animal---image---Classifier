import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

def setup_feature_extractor():
    """Set up ResNet50 model for feature extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, device

def extract_features(image_path, transform, feature_extractor, device):
    """Extract features from an image using ResNet50."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = feature_extractor(image)
        
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_data(base_dir, transform, feature_extractor, device, class_mapping):
    """Load and prepare data for training."""
    features = []
    labels = []
    
    for animal in tqdm(os.listdir(base_dir), desc="Loading data"):
        if animal not in class_mapping:
            continue
            
        animal_dir = os.path.join(base_dir, animal)
        if not os.path.isdir(animal_dir):
            continue
        
        for img_name in os.listdir(animal_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(animal_dir, img_name)
            img_features = extract_features(img_path, transform, feature_extractor, device)
            
            if img_features is not None:
                features.append(img_features)
                labels.append(class_mapping[animal])
    
    return np.array(features), np.array(labels)

def evaluate_model(y_true, y_pred, class_mapping):
    """Evaluate model performance and print detailed metrics."""
    # Get accuracy per class
    class_accuracies = {}
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
    for class_idx in range(len(class_mapping)):
        mask = (y_true == class_idx)
        if np.any(mask):
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            class_accuracies[reverse_mapping[class_idx]] = class_acc * 100
    
    # Sort classes by accuracy
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    # Print overall accuracy
    overall_acc = accuracy_score(y_true, y_pred) * 100
    print(f"\nOverall Accuracy: {overall_acc:.2f}%\n")
    
    # Print per-class accuracy
    print("Per-class Accuracy:")
    print("-" * 30)
    for class_name, acc in sorted_classes:
        print(f"{class_name:15s}: {acc:.2f}%")
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, 
                              target_names=list(class_mapping.keys()), 
                              digits=3))

def main():
    # Configuration
    base_dir = 'pro - 1/dataset/test'
    model_save_path = 'pro - 1/models/svm_model_retrained.joblib'
    scaler_save_path = 'pro - 1/models/scaler_retrained.joblib'
    
    # Class mapping
    class_mapping = {
        'Bear': 0, 'Bird': 1, 'Cat': 2, 'Cow': 3, 'Deer': 4,
        'Dog': 5, 'Dolphin': 6, 'Elephant': 7, 'Giraffe': 8,
        'Horse': 9, 'Kangaroo': 10, 'Lion': 11, 'Panda': 12,
        'Tiger': 13, 'Zebra': 14
    }
    
    # Set up transforms and feature extractor
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("Setting up feature extractor...")
    feature_extractor, device = setup_feature_extractor()
    
    # Load and prepare data
    print("Loading and preparing data...")
    features, labels = load_data(base_dir, transform, feature_extractor, device, class_mapping)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train model
    print("Training SVM model...")
    model = SVC(kernel='rbf', probability=True, class_weight='balanced')
    model.fit(features_scaled, labels)
    
    # Save model and scaler
    print("Saving model and scaler...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    # Evaluate model
    print("Evaluating model...")
    predictions = model.predict(features_scaled)
    evaluate_model(labels, predictions, class_mapping)

if __name__ == "__main__":
    main() 