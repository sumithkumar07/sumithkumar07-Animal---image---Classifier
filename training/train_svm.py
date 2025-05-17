import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import optuna
import joblib
import os
from datetime import datetime
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SVMOptimizer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Create output directory for models
        self.output_dir = 'models'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Keep track of best model
        self.best_score = 0
        self.best_model = None
        self.best_params = None
        
        # Progress tracking
        self.pbar = None
    
    def objective(self, trial):
        # Define hyperparameters to optimize
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'degree': trial.suggest_int('degree', 2, 5),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
        }
        
        # Create and train SVM model
        svm = SVC(**params, random_state=42, probability=True)
        
        # Use cross-validation to evaluate model
        scores = cross_val_score(svm, self.X_train_scaled, self.y_train, cv=5, scoring='accuracy')
        accuracy = scores.mean()
        
        # Update progress bar
        if self.pbar:
            self.pbar.update(1)
            self.pbar.set_postfix({'Best Accuracy': max(self.best_score, accuracy)})
        
        # Save if best model so far
        if accuracy > self.best_score:
            self.best_score = accuracy
            # Train on full training set
            svm.fit(self.X_train_scaled, self.y_train)
            self.best_model = svm
            self.best_params = params
            
            # Save model and scaler
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.output_dir, f'svm_model_{timestamp}.joblib')
            scaler_path = os.path.join(self.output_dir, f'scaler_{timestamp}.joblib')
            joblib.dump(svm, model_path)
            joblib.dump(self.scaler, scaler_path)
        
        return accuracy
    
    def optimize(self, n_trials=50):
        # Create study object
        study = optuna.create_study(direction='maximize')
        
        # Create progress bar
        self.pbar = tqdm(total=n_trials, desc="Optimizing SVM")
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials)
        
        # Close progress bar
        self.pbar.close()
        
        # Print optimization results
        print("\nOptimization Results:")
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Evaluate best model on test set
        y_pred = self.best_model.predict(self.X_test_scaled)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print("\nTest Set Performance:")
        print(f"Accuracy: {test_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return self.best_model, self.scaler, self.best_params

def load_data():
    """Load features and labels."""
    try:
        features = np.load('models/features.npy')
        labels = np.load('models/labels.npy')
        
        logger.info(f"Loaded features shape: {features.shape}")
        logger.info(f"Loaded labels shape: {labels.shape}")
        
        return features, labels
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_svm_model():
    # Load features and labels
    features, labels = load_data()
    class_mapping = np.load('models/class_mapping.npy', allow_pickle=True).item()
    
    print("Dataset Info:")
    print(f"Feature shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    print("\nClass mapping:")
    for class_name, idx in class_mapping.items():
        print(f"{class_name}: {idx}")
    
    # Reshape features if needed
    features = features.reshape(features.shape[0], -1)
    
    # Create and run optimizer
    optimizer = SVMOptimizer(features, labels)
    best_model, scaler, best_params = optimizer.optimize(n_trials=50)
    
    return best_model, scaler, best_params, class_mapping

def main():
    """Main function to run the training pipeline."""
    try:
        logger.info("Starting training pipeline...")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Train model
        model, scaler, best_params, class_mapping = train_svm_model()
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 