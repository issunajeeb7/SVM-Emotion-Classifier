import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm
import joblib

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('emotions (1).csv')

# Take a smaller sample for faster training
sample_size = 20000  # Using 20K samples for reasonable training time
df = df.sample(n=sample_size, random_state=42)

# Assuming the columns are 'text' and 'label'
X = df['text']
y = df['label']

# Convert labels to emotion names for better interpretation
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
y = y.map(emotion_map)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization using TF-IDF with reduced features
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(
    max_features=2000,  # Increased features for better performance
    stop_words='english',
    ngram_range=(1, 2)  # Using both unigrams and bigrams
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to train and evaluate SVM with different kernels
def train_evaluate_svm(kernel_name):
    print(f"\nTraining SVM with {kernel_name} kernel...")
    start_time = time.time()
    
    # Initialize SVM with appropriate parameters
    if kernel_name == 'linear':
        svm = SVC(kernel='linear', random_state=42, cache_size=2000)
    elif kernel_name == 'rbf':
        svm = SVC(kernel='rbf', random_state=42, cache_size=2000)
    elif kernel_name == 'poly':
        svm = SVC(kernel='poly', degree=3, random_state=42, cache_size=2000)
    else:  # sigmoid
        svm = SVC(kernel='sigmoid', random_state=42, cache_size=2000)
    
    # Train the model
    svm.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Print detailed classification report
    print(f"\nClassification Report for {kernel_name} kernel:")
    print(classification_report(y_test, y_pred))
    print(f"Training time: {training_time:.2f} seconds")
    
    return {
        'model': svm,
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'training_time': training_time
    }

# Train and evaluate different kernels
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
results = {}

print("\nTraining models with different kernels...")
for kernel in tqdm(kernels):
    results[kernel] = train_evaluate_svm(kernel)

# Print comparison table
print("\nComparison of Different Kernels:")
print("\n{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
    "Kernel", "Accuracy", "Precision", "Recall", "F1-Score", "Time (s)"))
print("-" * 60)

for kernel in kernels:
    print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.2f}".format(
        kernel,
        results[kernel]['accuracy'],
        results[kernel]['precision'],
        results[kernel]['recall'],
        results[kernel]['f1'],
        results[kernel]['training_time']
    ))

# Plot confusion matrix for the best performing kernel
best_kernel = max(results.keys(), key=lambda k: results[k]['f1'])
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, results[best_kernel]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emotion_map.values(),
            yticklabels=emotion_map.values())
plt.title(f'Confusion Matrix - {best_kernel.capitalize()} Kernel')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("\nModel training and evaluation complete!")
print(f"Best performing kernel: {best_kernel}")
print("Check confusion_matrix.png for visualization of the results.")

# Save the best model and vectorizer
joblib.dump(results[best_kernel]['model'], 'best_emotion_classifier_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("Best model and vectorizer saved as 'best_emotion_classifier_model.joblib' and 'tfidf_vectorizer.joblib'") 