import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import joblib
import os
import glob

class AnimalPredictor:
    def __init__(self, model_dir='models'):
        # Load the most recent model and scaler
        model_files = glob.glob(os.path.join(model_dir, 'svm_model_*.joblib'))
        scaler_files = glob.glob(os.path.join(model_dir, 'scaler_*.joblib'))
        
        if not model_files or not scaler_files:
            raise ValueError("No model or scaler found in models directory")
        
        # Get the most recent files
        latest_model = max(model_files, key=os.path.getctime)
        latest_scaler = max(scaler_files, key=os.path.getctime)
        
        # Load SVM model and scaler
        self.svm_model = joblib.load(latest_model)
        self.scaler = joblib.load(latest_scaler)
        
        # Load class mapping
        self.class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
        self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
        
        # Set up ResNet50 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = models.resnet50(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image)
        
        # Process features
        features = features.squeeze().cpu().numpy()
        features = features.reshape(1, -1)
        
        return features
    
    def predict(self, image_path):
        # Extract features
        features = self.extract_features(image_path)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.svm_model.predict(features_scaled)[0]
        probabilities = self.svm_model.predict_proba(features_scaled)[0]
        
        # Get top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        results = []
        
        for idx in top3_idx:
            animal_class = self.idx_to_class[idx]
            probability = probabilities[idx]
            results.append((animal_class, probability))
        
        return results

def main():
    # Example usage
    predictor = AnimalPredictor()
    
    # Test on a few images
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"Please create a '{test_dir}' directory and add some test images")
        return
    
    for image_file in os.listdir(test_dir):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, image_file)
            print(f"\nPredicting for image: {image_file}")
            
            predictions = predictor.predict(image_path)
            print("Top 3 predictions:")
            for animal_class, probability in predictions:
                print(f"{animal_class}: {probability:.2%}")

if __name__ == "__main__":
    main() 