import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor with ResNet50."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load pre-trained ResNet50
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Feature extractor initialized successfully")
    
    def extract_features(self, image_path):
        """Extract features from a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Transform and add batch dimension
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # Convert to numpy array
            features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {str(e)}")
            raise
    
    def process_dataset(self):
        """Extract features from all images in the processed dataset."""
        processed_dir = 'dataset/processed'
        
        # Load class mapping
        class_mapping = np.load('models/class_mapping.npy', allow_pickle=True).item()
        
        features_list = []
        labels_list = []
        
        # Process each class
        for class_name in sorted(os.listdir(processed_dir)):
            class_dir = os.path.join(processed_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            logger.info(f"Processing class: {class_name}")
            
            # Get all images in the class directory
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            # Process each image
            for image_name in tqdm(images, desc=f"Extracting features for {class_name}"):
                try:
                    image_path = os.path.join(class_dir, image_name)
                    features = self.extract_features(image_path)
                    
                    features_list.append(features)
                    labels_list.append(class_mapping[class_name])
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    continue
        
        # Convert lists to numpy arrays
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        # Save features and labels
        np.save('models/features.npy', features_array)
        np.save('models/labels.npy', labels_array)
        
        logger.info(f"Extracted features shape: {features_array.shape}")
        logger.info(f"Labels shape: {labels_array.shape}")
        
        return features_array, labels_array

def main():
    """Main function to run the feature extraction pipeline."""
    try:
        logger.info("Starting feature extraction pipeline...")
        
        # Initialize feature extractor
        extractor = FeatureExtractor()
        
        # Process dataset
        features, labels = extractor.process_dataset()
        
        logger.info("Feature extraction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in feature extraction pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 