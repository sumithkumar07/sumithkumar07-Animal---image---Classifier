import os
import requests
from PIL import Image
from io import BytesIO
import time
from tqdm import tqdm
import random
import logging
import cv2
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_download.log'),
        logging.StreamHandler()
    ]
)

def setup_directories(animals, base_dir='pro - 1/dataset/test'):
    """Create directories for each animal if they don't exist."""
    for animal in animals:
        os.makedirs(os.path.join(base_dir, animal), exist_ok=True)

def get_specific_queries(animal):
    """Get specific queries for challenging animal classes."""
    specific_queries = {
        'Dog': [
            'german shepherd dog breed',
            'golden retriever dog breed',
            'labrador dog breed',
            'husky dog breed',
            'bulldog breed',
            'dog portrait clear',
            'dog face closeup'
        ],
        'Horse': [
            'thoroughbred horse breed',
            'arabian horse breed',
            'horse portrait side view',
            'horse face closeup',
            'wild horse standing',
            'horse running profile'
        ],
        'Cat': [
            'domestic cat portrait',
            'cat face closeup',
            'house cat sitting',
            'cat profile view',
            'cat breed portrait',
            'cat clear photo'
        ],
        'Elephant': [
            'african elephant full body',
            'asian elephant side view',
            'elephant portrait closeup',
            'elephant walking profile',
            'elephant herd photo',
            'elephant trunk raised'
        ],
        'Zebra': [
            'zebra stripes clear',
            'zebra standing profile',
            'zebra face closeup',
            'zebra herd plains',
            'zebra full body photo',
            'zebra side view'
        ]
    }
    
    if animal in specific_queries:
        return specific_queries[animal]
    return [f"{animal} animal photo"]

def is_valid_image(file_path, min_size=(200, 200), max_size=(4000, 4000)):
    """Validate image file with stricter criteria for challenging classes."""
    try:
        with Image.open(file_path) as img:
            try:
                img.load()
            except Exception:
                return False
            
            # Check image dimensions
            if (img.size[0] < min_size[0] or img.size[1] < min_size[1] or 
                img.size[0] > max_size[0] or img.size[1] > max_size[1]):
                return False
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array for analysis
            img_array = np.array(img)
            
            # Check for standard deviation in each channel (color variation)
            std_per_channel = np.std(img_array, axis=(0,1))
            if np.any(std_per_channel < 20):  # Increased threshold for better quality
                return False
            
            # Check image sharpness using Laplacian variance
            if cv2.Laplacian(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() < 100:  # Increased threshold
                return False
            
            # Check for reasonable aspect ratio
            aspect_ratio = img.size[0] / img.size[1]
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return False
            
            return True
    except Exception:
        return False

def download_image(url, save_path):
    """Download and save an image from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
        img = img.resize((224, 224))
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading image: {e}")
        return False

def get_wikimedia_images(animal, num_images=50):
    """Get image URLs from Wikimedia Commons."""
    base_url = "https://commons.wikimedia.org/w/api.php"
    urls = []
    
    queries = get_specific_queries(animal)
    
    for query in queries:
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": f"{query}",
                "srnamespace": "6",  # File namespace
                "srlimit": num_images
            }
            
            response = requests.get(base_url, params=params)
            data = response.json()
            
            for item in data.get("query", {}).get("search", []):
                title = item["title"]
                if any(ext in title.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    file_name = title.replace("File:", "").replace(" ", "_")
                    image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{file_name}"
                    if image_url not in urls:  # Avoid duplicates
                        urls.append(image_url)
                    
            if len(urls) >= num_images:
                break
                
            time.sleep(0.5)  # Be nice to the server
                
        except Exception as e:
            print(f"Error fetching Wikimedia images for query '{query}': {e}")
            continue
    
    return urls[:num_images]

def download_animal_images(animals, num_images=50):
    """Download images for each animal."""
    base_dir = 'pro - 1/dataset/test'
    setup_directories(animals, base_dir)
    
    for animal in animals:
        animal_dir = os.path.join(base_dir, animal)
        existing_images = len([f for f in os.listdir(animal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if existing_images >= num_images:
            print(f"\nSkipping {animal} - already has {existing_images} images")
            continue
            
        print(f"\nDownloading images for {animal} (need {num_images - existing_images} more)...")
        image_count = existing_images
        
        # Get image URLs from Wikimedia
        urls = get_wikimedia_images(animal, (num_images - existing_images) * 2)  # Get extra URLs as backup
        
        with tqdm(total=num_images - existing_images, desc=f"Downloading {animal} images") as pbar:
            for url in urls:
                if image_count >= num_images:
                    break
                    
                save_path = os.path.join(base_dir, animal, f"{animal}_{image_count}.jpg")
                if download_image(url, save_path):
                    if is_valid_image(save_path):
                        image_count += 1
                        pbar.update(1)
                    else:
                        try:
                            os.remove(save_path)
                        except:
                            pass
                
                time.sleep(0.5)  # Be nice to the server
        
        print(f"Downloaded {image_count - existing_images} new images for {animal} (total: {image_count})")

def main():
    # List of animals from our model
    animals = [
        'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
        'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra'
    ]
    
    # Download images
    download_animal_images(animals, num_images=50)

if __name__ == "__main__":
    main() 