from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.test_model import AnimalClassifier
import time
from PIL import Image
import io
import logging
import traceback
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
MODELS_FOLDER = os.path.join(ROOT_DIR, 'models')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the classifier
try:
    classifier = AnimalClassifier(models_dir=MODELS_FOLDER)
except Exception as e:
    logger.error(f"Failed to initialize classifier: {str(e)}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    raise

# Supported image formats
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(file):
    file.seek(0, os.SEEK_END)
    size_bytes = file.tell()
    file.seek(0)  # Reset file pointer
    return size_bytes / (1024 * 1024)

def validate_image(file_stream):
    try:
        # Check file size
        file_size_mb = get_file_size_mb(file_stream)
        if file_size_mb > 15:  # Leave some margin below 16MB limit
            return False, f"File size too large: {file_size_mb:.1f}MB (maximum 15MB)"
        
        # Try to open the image with PIL
        image = Image.open(file_stream)
        
        # Check if image is actually an image
        try:
            image.verify()
            file_stream.seek(0)  # Reset file pointer
            image = Image.open(file_stream)
        except Exception as e:
            return False, "Invalid or corrupted image file"
        
        # Check image dimensions
        if image.size[0] < 10 or image.size[1] < 10:
            return False, f"Image too small: {image.size[0]}x{image.size[1]} pixels (minimum 10x10)"
        if image.size[0] > 4096 or image.size[1] > 4096:
            return False, f"Image too large: {image.size[0]}x{image.size[1]} pixels (maximum 4096x4096)"
        
        # Check and convert image mode
        if image.mode not in ('RGB', 'RGBA'):
            try:
                image = image.convert('RGB')
            except Exception as e:
                return False, f"Unable to convert image to RGB format: {str(e)}"
        
        logger.info(f"Image validated successfully: {image.size[0]}x{image.size[1]} pixels, {file_size_mb:.1f}MB, mode: {image.mode}")
        return True, image
        
    except Exception as e:
        logger.error(f"Error validating image: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return False, "Error validating image file"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            })
        
        # Validate image
        is_valid, result = validate_image(file)
        if not is_valid:
            return jsonify({'error': result})
        
        # Save the validated and potentially converted image
        filename = secure_filename(f"{int(time.time())}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            if isinstance(result, Image.Image):
                # Save the PIL Image
                result.save(filepath, format='PNG')
            else:
                # Save the original file
                file.save(filepath)
            
            # Get predictions
            try:
                results = classifier.predict(filepath)
                
                if not results:
                    return jsonify({'error': 'No predictions generated. Please try a different image.'})
                
                # Format results
                predictions = [{'class': animal_class, 'probability': float(prob)} 
                             for animal_class, prob in results]
                
                # Log the top prediction
                top_pred = max(predictions, key=lambda x: x['probability'])
                logger.info(f"Top prediction for {filename}: {top_pred['class']} ({top_pred['probability']*100:.2f}%)")
                
                return jsonify({
                    'success': True,
                    'predictions': predictions
                })
                
            except ValueError as ve:
                # Handle known validation errors
                return jsonify({'error': str(ve)})
            except Exception as e:
                # Handle unexpected errors during prediction
                logger.error(f"Error during prediction: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                return jsonify({'error': 'Error analyzing image. Please try a different image.'})
            
        except Exception as e:
            logger.error(f"Error saving or processing image: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return jsonify({'error': 'Error processing image. Please try again.'})
            
        finally:
            # Clean up temporary file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred. Please try again.'})

if __name__ == '__main__':
    app.run(debug=True, port=5000) 