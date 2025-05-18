import joblib
import os
import glob
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_best_model():
    # Find the most recent model and scaler
    model_files = glob.glob(os.path.join('models', 'svm_model_*.joblib'))
    if not model_files:
        print("No model files found!")
        return
    
    latest_model_path = max(model_files, key=os.path.getctime)
    latest_model = joblib.load(latest_model_path)
    
    # Load test data
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    class_mapping = np.load('class_mapping.npy', allow_pickle=True).item()
    
    # Get model parameters
    params = latest_model.get_params()
    
    # Print model information
    print("\n=== Best Model Hyperparameters ===")
    important_params = ['C', 'gamma', 'kernel', 'degree', 'class_weight']
    for param in important_params:
        print(f"{param}: {params[param]}")
    
    # Load scaler
    scaler_files = glob.glob(os.path.join('models', 'scaler_*.joblib'))
    latest_scaler_path = max(scaler_files, key=os.path.getctime)
    scaler = joblib.load(latest_scaler_path)
    
    # Prepare data
    features_reshaped = features.reshape(features.shape[0], -1)
    features_scaled = scaler.transform(features_reshaped)
    
    # Get predictions
    predictions = latest_model.predict(features_scaled)
    probabilities = latest_model.predict_proba(features_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\n=== Classification Report ===")
    class_names = [name for name, idx in sorted(class_mapping.items(), key=lambda x: x[1])]
    print(classification_report(labels, predictions, target_names=class_names))
    
    # Create confusion matrix visualization
    plt.figure(figsize=(15, 10))
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate and display class-wise accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = labels == i
        class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
        class_accuracy[class_name] = class_acc
    
    # Plot class-wise accuracy
    plt.figure(figsize=(15, 6))
    classes = list(class_accuracy.keys())
    accuracies = list(class_accuracy.values())
    plt.bar(classes, accuracies)
    plt.title('Class-wise Accuracy')
    plt.xlabel('Animal Class')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_accuracy.png')
    plt.close()
    
    # Print class-wise accuracy
    print("\n=== Class-wise Accuracy ===")
    for class_name, acc in class_accuracy.items():
        print(f"{class_name}: {acc:.4f}")
    
    return params, accuracy, class_accuracy

if __name__ == "__main__":
    # Add required import here to avoid confusion with the early import
    from sklearn.metrics import confusion_matrix
    analyze_best_model() 