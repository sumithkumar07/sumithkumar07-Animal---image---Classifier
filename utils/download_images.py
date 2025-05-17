import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from PIL import Image
import hashlib
import requests
import logging
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import quote

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_download.log'),
        logging.StreamHandler()
    ]
)

def download_image(url, save_path, max_retries=2):
    """Download a single image."""
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            with requests.get(url, timeout=10, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['jpeg', 'jpg', 'png']):
                        return False
                    
                    ext = '.jpg' if ('jpeg' in content_type or 'jpg' in content_type) else '.png'
                    base_path = os.path.splitext(save_path)[0]
                    save_path = base_path + ext
                    
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
    return False

def is_valid_image(file_path, min_size=(200, 200), max_size=(4000, 4000)):
    """Validate image file."""
    try:
        with Image.open(file_path) as img:
            try:
                img.load()
            except Exception:
                return False
            
            if (img.size[0] < min_size[0] or img.size[1] < min_size[1] or 
                img.size[0] > max_size[0] or img.size[1] > max_size[1]):
                return False
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            std_per_channel = np.std(img_array, axis=(0,1))
            if np.any(std_per_channel < 10):
                return False
            
            if cv2.Laplacian(np.array(img), cv2.CV_64F).var() < 50:
                return False
            
            return True
    except Exception:
        return False

def get_image_hash(file_path):
    """Generate MD5 hash of image file."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def clean_dataset_folder(folder_path):
    """Remove invalid or duplicate images from folder."""
    logging.info(f"Cleaning folder: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    seen_hashes = set()
    removed = 0
    
    for filename in tqdm(files, desc="Cleaning images"):
        file_path = os.path.join(folder_path, filename)
        
        if not is_valid_image(file_path):
            os.remove(file_path)
            removed += 1
            continue
        
        img_hash = get_image_hash(file_path)
        if img_hash in seen_hashes:
            os.remove(file_path)
            removed += 1
        else:
            seen_hashes.add(img_hash)
    
    logging.info(f"Removed {removed} invalid or duplicate images")

def get_bing_image_urls(query, count=150):
    """Get image URLs directly from Bing Image Search API."""
    urls = set()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.bing.com/images/search'
    }
    
    offsets = [0, 50, 100, 150]  # Get images from different offsets
    for offset in offsets:
        try:
            params = {
                'q': quote(query),
                'first': offset,
                'count': min(50, count - len(urls)),
                'qft': '+filterui:photo-photo+filterui:imagesize-large',
                'adlt': 'moderate'
            }
            
            url = 'https://www.bing.com/images/async'
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                # Extract image URLs from response
                image_urls = response.text.split('murl&quot;:&quot;')[1:]
                for img_url in image_urls:
                    try:
                        url = img_url.split('&quot;')[0]
                        if url.startswith('http'):
                            urls.add(url)
                    except:
                        continue
            
            if len(urls) >= count:
                break
                
            time.sleep(1)
            
        except Exception:
            continue
    
    return list(urls)

def concurrent_download_images(urls, save_dir, animal_class, needed_count):
    """Download images concurrently."""
    successful_downloads = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {}
        for i, url in enumerate(urls):
            if successful_downloads >= needed_count:
                break
            
            filename = f"{animal_class}_{int(time.time())}_{i}"
            save_path = os.path.join(save_dir, filename)
            
            future = executor.submit(download_image, url, save_path)
            future_to_url[future] = (url, save_path)
        
        with tqdm(total=needed_count, desc=f"Downloading {animal_class} images") as pbar:
            for future in as_completed(future_to_url):
                url, save_path = future_to_url[future]
                try:
                    if future.result():
                        if is_valid_image(save_path):
                            successful_downloads += 1
                            pbar.update(1)
                        else:
                            try:
                                os.remove(save_path)
                            except:
                                pass
                except Exception:
                    try:
                        os.remove(save_path)
                    except:
                        pass
                
                if successful_downloads >= needed_count:
                    break
    
    return successful_downloads

def get_diverse_queries(animal_class):
    """Generate diverse search queries for each animal class."""
    base_queries = {
        'Bear': ['black bear', 'grizzly bear', 'polar bear', 'brown bear', 'panda bear'],
        'Bird': ['eagle', 'parrot', 'owl', 'sparrow', 'peacock', 'hummingbird'],
        'Cat': ['lion', 'tiger', 'leopard', 'cheetah', 'house cat', 'wild cat'],
        'Cow': ['dairy cow', 'beef cattle', 'highland cow', 'farm cow', 'cattle'],
        'Deer': ['whitetail deer', 'red deer', 'mule deer', 'reindeer', 'elk'],
        'Dog': ['golden retriever', 'german shepherd', 'labrador', 'husky', 'bulldog'],
        'Dolphin': ['bottlenose dolphin', 'orca', 'spinner dolphin', 'spotted dolphin'],
        'Elephant': ['african elephant', 'asian elephant', 'baby elephant', 'wild elephant'],
        'Giraffe': ['masai giraffe', 'reticulated giraffe', 'rothschild giraffe'],
        'Horse': ['arabian horse', 'wild horse', 'mustang horse', 'thoroughbred'],
        'Kangaroo': ['red kangaroo', 'grey kangaroo', 'wallaby', 'joey kangaroo'],
        'Lion': ['male lion', 'lioness', 'african lion', 'pride of lions', 'lion cubs'],
        'Panda': ['giant panda', 'red panda', 'baby panda', 'wild panda'],
        'Tiger': ['bengal tiger', 'siberian tiger', 'white tiger', 'sumatran tiger'],
        'Zebra': ['plains zebra', 'mountain zebra', 'grevy zebra', 'zebra herd']
    }
    
    specific_queries = base_queries.get(animal_class, [animal_class])
    queries = []
    
    for specific in specific_queries:
        queries.extend([
            f"{specific} animal photo",
            f"{specific} in wild",
            f"{specific} high resolution",
            f"{specific} wildlife photography"
        ])
    
    return list(set(queries))

def download_images_for_class(animal_class, target_count=200):
    """Download images for a specific animal class."""
    logging.info(f"Starting download for: {animal_class}")
    
    save_path = os.path.join('dataset', animal_class)
    os.makedirs(save_path, exist_ok=True)
    
    clean_dataset_folder(save_path)
    
    existing_count = len([f for f in os.listdir(save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if existing_count >= target_count:
        logging.info(f"Already have {existing_count} images for {animal_class}")
        return
    
    needed_count = target_count - existing_count
    logging.info(f"Need to download {needed_count} more images")
    
    search_queries = get_diverse_queries(animal_class)
    random.shuffle(search_queries)
    
    all_urls = set()
    for query in search_queries:
        if len(all_urls) >= needed_count * 2:  # Get extra URLs for backup
            break
        
        try:
            urls = get_bing_image_urls(query)
            all_urls.update(urls)
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error processing query '{query}': {str(e)}")
            continue
    
    url_list = list(all_urls)
    random.shuffle(url_list)
    
    downloaded = concurrent_download_images(url_list, save_path, animal_class, needed_count)
    
    clean_dataset_folder(save_path)
    
    final_count = len([f for f in os.listdir(save_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    new_images = final_count - existing_count
    logging.info(f"Downloaded {new_images} new images. Total images for {animal_class}: {final_count}")

def main():
    """Main function to download images for all animal classes."""
    animal_classes = [
        'Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant',
        'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra'
    ]
    
    for animal in animal_classes:
        try:
            current_count = len([f for f in os.listdir(os.path.join('dataset', animal)) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            if current_count >= 200:
                logging.info(f"Skipping {animal} - already has {current_count} images")
                continue
                
            download_images_for_class(animal)
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Error processing animal class {animal}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 