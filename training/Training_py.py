#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import re
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
from tqdm import tqdm
import os

# Download NLTK resources (only once)
# nltk.download('stopwords')
# nltk.download('punkt')

def train_model():
    # Load dataset
    data = pd.read_csv('Training_Data.csv')

    # Create labels based on ratings
    pos_neg = [1 if rating >= 3 else 0 for rating in data['rating']]
    data['label'] = pos_neg

    # Preprocess text
    def preprocess_text(text_data):
        preprocessed_text = []
        for sentence in tqdm(text_data):
            sentence = re.sub(r'[^\w\s]', '', sentence)
            preprocessed_text.append(
                ' '.join(
                    token.lower() for token in nltk.word_tokenize(sentence)
                    if token.lower() not in stopwords.words('english')
                )
            )
        return preprocessed_text

    data['review'] = preprocess_text(data['review'].values)

    # Generate WordCloud (optional)
    consolidated = ' '.join(word for word in data['review'][data['label'] == 1].astype(str))
    wordCloud = WordCloud(width=1600, height=800, random_state=21, max_font_size=110)
    plt.figure(figsize=(15, 10))
    plt.imshow(wordCloud.generate(consolidated), interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Create TF-IDF features
    cv = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X = cv.fit_transform(data['review']).toarray()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, data['label'], test_size=0.33, stratify=data['label'], random_state=42
    )

    # Train the model
    model = DecisionTreeClassifier(random_state=0, class_weight='balanced')
    model.fit(X_train, y_train)

    # Evaluate the model
    pred = model.predict(X_train)
    print(f"Training Accuracy: {accuracy_score(y_train, pred)}")
    test_pred = model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, test_pred)}")

    # Confusion Matrix (optional)
    cm = confusion_matrix(y_train, pred)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
    cm_display.plot()
    plt.show()

    # Save the model and vectorizer in the same directory as this script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(cv, f)

    with open(os.path.join(base_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("✅ Vectorizer and model saved successfully!")

# Only run if script is directly executed (not imported)
if __name__ == "__main__":
    train_model()
