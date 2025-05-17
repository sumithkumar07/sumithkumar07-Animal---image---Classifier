from setuptools import setup, find_packages

setup(
    name="animal-classifier",
    version="1.0.0",
    description="A deep learning-based system for classifying animals using ResNet50 and SVM",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'optuna>=3.0.0',
        'joblib>=1.1.0',
        'Pillow>=9.0.0',
        'tqdm>=4.65.0',
        'flask>=2.0.0',
        'flask-cors>=4.0.0',
        'requests>=2.28.0',
        'python-dotenv>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
        'albumentations>=1.3.0',
        'opencv-python>=4.7.0'
    ],
    python_requires='>=3.8',
) 