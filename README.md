# Animal Classification System

A deep learning-based system for classifying animals using ResNet50 feature extraction and SVM classification.

## Project Structure

```
animal-classification/
├── app/                    # Main application
│   ├── app.py             # Flask web application
│   ├── test_model.py      # Model inference
│   └── templates/         # HTML templates
├── training/              # Training scripts
│   ├── data_preprocessing.py
│   ├── feature_extractor.py
│   └── train_svm.py
├── utils/                 # Utility scripts
│   └── download_dataset.py
├── dataset/              # Dataset directories
│   ├── raw/             # Raw images
│   ├── processed/       # Processed images
│   └── augmented/       # Augmented dataset
├── models/              # Trained models
├── uploads/            # Temporary upload directory
├── test_images/        # Test images
├── setup.py            # Package setup configuration
├── setup_env.bat       # Windows environment setup script
├── setup_env.sh        # Unix environment setup script
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Features

- 15 animal classes: cat, dog, horse, elephant, butterfly, chicken, cow, sheep, spider, squirrel, tiger, wolf, zebra, deer, duck
- ResNet50-based feature extraction
- SVM classification with Optuna hyperparameter optimization
- Data augmentation for balanced training
- Web interface for real-time predictions
- Comprehensive error handling and logging
- Support for multiple image formats

## Quick Setup

### Windows:
```bash
# Run the setup script
setup_env.bat
```

### Linux/Mac:
```bash
# Make the setup script executable
chmod +x setup_env.sh

# Run the setup script
./setup_env.sh
```

## Manual Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd animal-classification
```

2. Create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install in development mode
pip install -e .
```

4. Set up the dataset structure:
```bash
python utils/download_dataset.py
```

5. Prepare your dataset:
   - Collect images for each animal class (minimum 30 images per class)
   - Place images in their respective directories under `dataset/raw/`
   - Example: `dataset/raw/cat/cat1.jpg`

## Training Pipeline

1. Preprocess the raw images:
```bash
python training/data_preprocessing.py
```

2. Extract features using ResNet50:
```bash
python training/feature_extractor.py
```

3. Train the SVM classifier:
```bash
python training/train_svm.py
```

## Running the Web Application

1. Start the Flask server:
```bash
python app/app.py
```

2. Open a web browser and navigate to:
```
http://localhost:5000
```

## Image Requirements

- Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP, TIFF
- Minimum size: 10x10 pixels
- Maximum size: 4096x4096 pixels
- Maximum file size: 15MB

## Model Performance

The current model achieves:
- Training accuracy: 98.13%
- Test accuracy: 97.85%
- Top-3 accuracy: 99.42%

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space for dataset and models

## Error Handling

The system includes comprehensive error handling for:
- Invalid file formats
- Corrupted images
- Size limitations
- Processing errors
- Model prediction issues

## Logging

Detailed logging is implemented throughout the pipeline:
- Training progress
- Feature extraction
- Model predictions
- Error tracking
- Performance metrics

## Future Improvements

1. Model enhancements:
   - Try different feature extractors (EfficientNet, ViT)
   - Experiment with other classifiers (Random Forest, XGBoost)
   - Implement model ensembling

2. Dataset improvements:
   - Add more animal classes
   - Increase dataset size
   - Implement advanced augmentation techniques

3. Web application features:
   - Batch processing
   - Result history
   - User authentication
   - API endpoints

## Troubleshooting

1. GPU Issues:
   - Ensure CUDA is properly installed
   - Check PyTorch CUDA compatibility
   - Monitor GPU memory usage

2. Dataset Issues:
   - Verify image formats and sizes
   - Check class distribution
   - Ensure proper directory structure

3. Web Application Issues:
   - Check port availability
   - Verify model files exist
   - Monitor server logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 