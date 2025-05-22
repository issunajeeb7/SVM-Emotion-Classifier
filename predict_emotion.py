import joblib
import pandas as pd
import numpy as np

def load_model_and_vectorizer():
    """Load the saved model and vectorizer"""
    try:
        model = joblib.load('best_emotion_classifier_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        print("Error: Model files not found. Please make sure you have run emotion_classifier.py first.")
        return None, None

def predict_emotion(text, model, vectorizer):
    """Predict emotion for the given text"""
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vectorized)[0]
    
    # Get prediction probabilities
    try:
        probabilities = model.predict_proba(text_vectorized)[0]
    except:
        probabilities = None
    
    return prediction, probabilities

def main():
    # Load the model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    
    if model is None or vectorizer is None:
        return
    
    # Emotion mapping
    emotion_map = {
        'sadness': 0,
        'joy': 1,
        'love': 2,
        'anger': 3,
        'fear': 4,
        'surprise': 5
    }
    
    # Reverse mapping for display
    reverse_emotion_map = {v: k for k, v in emotion_map.items()}
    
    print("\nEmotion Prediction System")
    print("------------------------")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input from user
        text = input("\nEnter your text (or 'quit' to exit): ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            print("Please enter some text!")
            continue
        
        # Make prediction
        prediction, probabilities = predict_emotion(text, model, vectorizer)
        
        # Display results
        print("\nPrediction Results:")
        print(f"Predicted Emotion: {prediction}")
        
        if probabilities is not None:
            print("\nEmotion Probabilities:")
            for emotion, prob in zip(reverse_emotion_map.values(), probabilities):
                print(f"{emotion}: {prob:.2%}")

if __name__ == "__main__":
    main() 