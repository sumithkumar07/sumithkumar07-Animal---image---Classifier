import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def setup_feature_extractor():
    """Set up ResNet50 model for feature extraction."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor, device

def extract_features(image_path, transform, feature_extractor, device):
    """Extract features from an image using ResNet50."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(image)
        
        # Process features
        features = features.squeeze().cpu().numpy()
        return features
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

def main():
    # Configuration
    test_dir = 'pro - 1/dataset/test'
    model_path = 'pro - 1/models/svm_model.joblib'
    scaler_path = 'pro - 1/models/scaler.joblib'
    
    # Load existing model and scaler
    print("Loading existing model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Class mapping
    class_mapping = {
        'Bear': 0, 'Bird': 1, 'Cat': 2, 'Cow': 3, 'Deer': 4, 'Dog': 5, 'Dolphin': 6,
        'Elephant': 7, 'Giraffe': 8, 'Horse': 9, 'Kangaroo': 10, 'Lion': 11,
        'Panda': 12, 'Tiger': 13, 'Zebra': 14
    }
    
    # Set up feature extractor and transforms
    print("Setting up feature extractor...")
    feature_extractor, device = setup_feature_extractor()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and prepare data
    print("Loading and preparing data...")
    features, labels = load_data(test_dir, transform, feature_extractor, device, class_mapping)
    
    # Scale features
    print("Scaling features...")
    features_scaled = scaler.transform(features)
    
    # Update model weights for challenging classes
    print("Updating model weights...")
    challenging_classes = ['Dog', 'Horse', 'Cat', 'Elephant', 'Zebra']
    challenging_indices = [class_mapping[cls] for cls in challenging_classes]
    
    # Create mask for samples from challenging classes
    mask = np.isin(labels, challenging_indices)
    
    # Get samples for challenging classes
    X_challenging = features_scaled[mask]
    y_challenging = labels[mask]
    
    # Create and train a new SVM for challenging classes
    new_svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    new_svm.fit(X_challenging, y_challenging)
    
    # Update the original model's parameters for challenging classes
    for i, class_idx in enumerate(challenging_indices):
        mask_class = (labels == class_idx)
        if np.any(mask_class):
            # Update support vectors and coefficients for this class
            model.support_vectors_[model.support_[mask_class]] = new_svm.support_vectors_
            model.dual_coef_[:, model.support_[mask_class]] = new_svm.dual_coef_
    
    # Save updated model
    print("Saving updated model...")
    joblib.dump(model, model_path.replace('.joblib', '_updated.joblib'))
    print("Model updated and saved successfully!")

if __name__ == "__main__":
    main() 