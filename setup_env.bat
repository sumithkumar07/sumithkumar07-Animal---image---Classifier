@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate

echo Installing dependencies...
python -m pip install --upgrade pip
pip install -e .

echo Creating necessary directories...
python utils\download_dataset.py

echo Setup complete!
echo.
echo Next steps:
echo 1. Collect images for each animal class (minimum 30 images per class)
echo 2. Place images in dataset/raw/[class_name] directories
echo 3. Run the training pipeline:
echo    - python training/data_preprocessing.py
echo    - python training/feature_extractor.py
echo    - python training/train_svm.py
echo 4. Start the web application:
echo    - python app/app.py
echo.
pause 