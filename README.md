# Emotion Classification using SVM

This project implements a machine learning model to classify emotions in English-language tweets using Support Vector Machines (SVM). The model can predict six basic emotions: sadness, joy, love, anger, fear, and surprise.

## Project Structure

- `emotion_classifier.py`: Main script for training the SVM model with different kernels
- `predict_emotion.py`: Interactive script for making predictions on new text
- `requirements.txt`: List of required Python packages
- `emotions (1).csv`: Dataset containing labeled tweets (not included in repository)

## Features

- Implements SVM with multiple kernels (linear, RBF, polynomial, sigmoid)
- Uses TF-IDF vectorization for text processing
- Provides detailed performance metrics for each kernel
- Interactive prediction system for new text input
- Saves the best performing model for future use

## Performance

The model was evaluated using different SVM kernels, with the following results:

| Kernel   | Accuracy | Precision | Recall | F1-Score | Time (s) |
|----------|----------|-----------|--------|----------|----------|
| linear   | 0.8852   | 0.8867    | 0.8852 | 0.8847   | 15.91    |
| rbf      | 0.8812   | 0.8829    | 0.8812 | 0.8789   | 38.61    |
| poly     | 0.7398   | 0.7922    | 0.7398 | 0.7177   | 41.57    |
| sigmoid  | 0.8840   | 0.8858    | 0.8840 | 0.8833   | 18.66    |

The linear kernel performed best in terms of both accuracy and training time.

## Setup and Usage

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python emotion_classifier.py
```

3. Make predictions:
```bash
python predict_emotion.py
```

## Model Details

- **Text Processing**: TF-IDF vectorization with 2000 features
- **Training Data**: 20,000 samples from the dataset
- **Test Split**: 20% of the data
- **Emotion Classes**: 6 emotions (sadness, joy, love, anger, fear, surprise)

## Files Generated

- `best_emotion_classifier_model.joblib`: Saved best performing model
- `tfidf_vectorizer.joblib`: Saved TF-IDF vectorizer
- `confusion_matrix.png`: Visualization of model performance

