import os
import sys
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import joblib
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_webp_to_rgb(image_path):
    """Convert webp image to RGB format."""
    try:
        with Image.open(image_path) as img:
            if img.format == 'WEBP':
                img = img.convert('RGB')
            return img
    except Exception as e:
        logger.error(f"Error converting image {image_path}: {str(e)}")
        return None

class ImageClassifier:
    def __init__(self):
        try:
            # Get the current script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(script_dir, 'models')
            
            # Load SVM model and scaler
            model_path = os.path.join(models_dir, 'svm_model.joblib')
            scaler_path = os.path.join(models_dir, 'scaler.joblib')
            mapping_path = os.path.join(models_dir, 'class_mapping.npy')
            
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            logger.info(f"Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load class mapping
            self.class_mapping = np.load(mapping_path, allow_pickle=True).item()
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            
            # Set up ResNet50
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            # Define transformations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing classifier: {str(e)}")
            raise
    
    def predict(self, image_path):
        try:
            # Load and preprocess image
            image = convert_webp_to_rgb(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(image)
            
            # Process features
            features = features.squeeze().cpu().numpy()
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get top 3 predictions
            top3_idx = np.argsort(probabilities)[-3:][::-1]
            results = []
            
            for idx in top3_idx:
                animal_class = self.idx_to_class[idx]
                probability = probabilities[idx]
                results.append((animal_class, probability))
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            raise

def analyze_test_images():
    """Analyze all test images and calculate accuracy."""
    try:
        # Initialize classifier
        classifier = ImageClassifier()
        logger.info("Classifier initialized successfully")
        
        # Get all images from test_images directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(script_dir, "test_images")
        
        image_files = [
            os.path.join(test_dir, f) 
            for f in os.listdir(test_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ]
        
        if not image_files:
            logger.error("No images found in test_images directory!")
            return
        
        logger.info(f"Found {len(image_files)} images to test")
        
        # Expected classes for each image type
        expected_classes = {
            'bear': ['brown_bear', 'polar_bear'],
            'deer': ['deer', 'fawn'],
            'cow': ['cow', 'cattle'],
            'cat': ['cat', 'feline'],
            'dog': ['dog', 'canine'],
            'dolphin': ['dolphin', 'marine_mammal'],
            'zebra': ['zebra', 'equine']
        }
        
        # Process each image
        correct_predictions = 0
        total_predictions = len(image_files)
        
        print("\n=== Testing Model Performance ===")
        for img_path in image_files:
            img_name = os.path.basename(img_path)
            print(f"\nProcessing: {img_name}")
            
            try:
                results = classifier.predict(img_path)
                
                print("Top 3 Predictions:")
                for animal_class, probability in results:
                    print(f"{animal_class}: {probability*100:.2f}%")
                
                # Check if top prediction matches expected class
                top_prediction = results[0][0].lower()
                
                # Determine if prediction is correct based on filename
                filename_lower = img_name.lower()
                is_correct = False
                
                for class_type, valid_classes in expected_classes.items():
                    if class_type in filename_lower:
                        is_correct = any(valid_class.lower() in top_prediction for valid_class in valid_classes)
                        break
                
                if is_correct:
                    correct_predictions += 1
                    print("✓ Correct prediction")
                else:
                    print("✗ Incorrect prediction")
                
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {str(e)}")
                total_predictions -= 1
                continue
        
        # Calculate and display accuracy
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"\n=== Overall Results ===")
            print(f"Total images tested: {total_predictions}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Test set accuracy: {accuracy:.2f}%")
            print(f"Benchmark accuracy: 98.13%")
            print(f"Difference: {accuracy - 98.13:.2f}%")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    analyze_test_images() 