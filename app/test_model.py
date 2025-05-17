import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import joblib
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import traceback
import sys

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnimalClassifier:
    def __init__(self, models_dir='models'):
        """Initialize the classifier with the trained model and necessary components."""
        try:
            self.models_dir = models_dir
            
            # Load SVM model
            model_path = os.path.join(models_dir, 'svm_model_20250517_231924.joblib')
            logger.info(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(models_dir, 'scaler_20250517_231924.joblib')
            logger.info(f"Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load class mapping
            mapping_path = os.path.join(models_dir, 'class_mapping.npy')
            self.class_mapping = np.load(mapping_path, allow_pickle=True).item()
            logger.info(f"Loaded {len(self.class_mapping)} classes from mapping")
            
            self.idx_to_class = {v: k for k, v in self.class_mapping.items()}
            
            # Set up ResNet50 model
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            try:
                self.feature_extractor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
                self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
                self.feature_extractor = self.feature_extractor.to(self.device)
                self.feature_extractor.eval()
            except Exception as e:
                logger.error(f"Error loading ResNet50 model: {str(e)}")
                raise
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Animal Classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Animal Classifier: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            image = Image.open(image_path)
            
            # Check image dimensions
            if image.size[0] < 10 or image.size[1] < 10:
                raise ValueError(f"Image too small: {image.size[0]}x{image.size[1]} (minimum 10x10)")
            if image.size[0] > 4096 or image.size[1] > 4096:
                raise ValueError(f"Image too large: {image.size[0]}x{image.size[1]} (maximum 4096x4096)")
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode not in ['RGB']:
                image = image.convert('RGB')
            
            logger.info(f"Successfully preprocessed image: {os.path.basename(image_path)} ({image.size[0]}x{image.size[1]}, mode: {image.mode})")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def extract_features(self, image):
        """Extract features using ResNet50."""
        try:
            # Additional input validation
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feature_extractor(image_tensor)
            
            features = features.squeeze().cpu().numpy()
            features = features.reshape(1, -1)
            
            if np.isnan(features).any():
                raise ValueError("Feature extraction produced NaN values")
            
            logger.info(f"Successfully extracted features from image (shape: {features.shape})")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def predict(self, image_path):
        """Predict animal class for an image."""
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            
            # Extract features
            features = self.extract_features(image)
            
            # Scale features
            try:
                features_scaled = self.scaler.transform(features)
            except Exception as e:
                logger.error(f"Error scaling features: {str(e)}")
                raise ValueError("Error scaling features. The image might be too different from the training data.")
            
            # Get predictions and probabilities
            try:
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
            except Exception as e:
                logger.error(f"Error during model prediction: {str(e)}")
                raise ValueError("Error making prediction. The model might not be compatible with the input.")
            
            # Get top 3 predictions
            top3_idx = np.argsort(probabilities)[-3:][::-1]
            results = []
            
            for idx in top3_idx:
                try:
                    animal_class = self.idx_to_class[idx]
                    probability = probabilities[idx]
                    results.append((animal_class, probability))
                except KeyError:
                    logger.error(f"Class index {idx} not found in mapping")
                    raise ValueError(f"Invalid class index {idx}. The model and class mapping might be mismatched.")
            
            logger.info(f"Successfully generated predictions for {os.path.basename(image_path)}")
            logger.info(f"Top prediction: {results[0][0]} ({results[0][1]:.2%})")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    
    def display_prediction(self, image_path):
        """Display image with predictions."""
        try:
            results = self.predict(image_path)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Display image
            plt.subplot(1, 2, 1)
            image = self.preprocess_image(image_path)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Input Image')
            
            # Display predictions
            plt.subplot(1, 2, 2)
            classes = [r[0] for r in results]
            probs = [r[1] for r in results]
            
            y_pos = np.arange(len(classes))
            plt.barh(y_pos, probs)
            plt.yticks(y_pos, classes)
            plt.xlabel('Probability')
            plt.title('Top 3 Predictions')
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'predictions_{timestamp}.png'
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            # Print results
            logger.info(f"\nPredictions for: {os.path.basename(image_path)}")
            logger.info("-" * 50)
            for animal_class, prob in results:
                logger.info(f"{animal_class}: {prob:.2%}")
            logger.info(f"\nVisualization saved as: {save_path}")
            
        except Exception as e:
            logger.error(f"Error displaying prediction: {str(e)}")
            raise

def main():
    try:
        # Initialize classifier
        classifier = AnimalClassifier()
        
        # Process all images in test_images directory
        test_dir = 'test_images'
        if not os.path.exists(test_dir):
            logger.error(f"Error: {test_dir} directory not found!")
            return
        
        test_images = [f for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not test_images:
            logger.warning(f"No images found in {test_dir}!")
            logger.info("Please add some images and run again.")
            logger.info("Supported formats: .jpg, .jpeg, .png")
            return
        
        logger.info(f"\nFound {len(test_images)} images to process...")
        
        for image_file in test_images:
            image_path = os.path.join(test_dir, image_file)
            classifier.display_prediction(image_path)
            logger.info("\n" + "="*50 + "\n")
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 