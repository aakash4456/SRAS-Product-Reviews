Okay, based on the provided GeeksforGeeks article about NLP and cross-referencing with the details of your SRAS project (from your report draft and codebase), here's a list of methods and techniques you've used in SRAS, categorized similarly to the article where applicable:

**I. Text Processing and Preprocessing (as used in SRAS)**

1.  **Tokenization:**
    *   **SRAS:** You used `nltk.word_tokenize` to divide review text into individual words (tokens). This is a fundamental first step.
2.  **Stop-word Removal:**
    *   **SRAS:** You removed common English words (like "and", "the", "is") using NLTK's `stopwords.words('english')` because they generally don't carry significant sentiment.
3.  **Text Normalization:**
    *   **Case Normalization (Lowercasing):**
        *   **SRAS:** You converted all text to lowercase (`token.lower()`) to treat words like "Good" and "good" as the same, ensuring consistency.
    *   **Punctuation/Special Character Removal:**
        *   **SRAS:** You used regular expressions (`re.sub(r'[^\w\s]', '', sentence)`) to remove characters that are not alphanumeric or whitespace, simplifying the text.

    *(Note: Stemming and Lemmatization were mentioned in the GfG article and your literature review as common techniques, but your SRAS implementation focuses on the above for preprocessing before TF-IDF).*

**II. Text Representation / Feature Extraction (as used in SRAS)**

1.  **Term Frequency-Inverse Document Frequency (TF-IDF):**
    *   **SRAS:** This is your core method for converting preprocessed text into numerical vectors. You used `sklearn.feature_extraction.text.TfidfVectorizer`.
        *   It reflects the importance of a word in a specific review relative to all reviews in your dataset.
        *   You configured it with `max_features=2500` (to select the top 2500 terms) and `ngram_range=(1, 2)` (to use both single words and two-word phrases as features).

    *(Note: Bag of Words (BoW) and Word Embeddings are other text representation techniques mentioned in the GfG article and your literature review, but TF-IDF is what you implemented).*

**III. Text Classification (as used in SRAS for Sentiment Analysis)**

1.  **Sentiment Analysis:**
    *   **SRAS:** This is the primary NLP task your project performs. You are determining the sentiment (Positive/Negative) expressed in the review text.
2.  **Machine Learning-Based Approach:**
    *   **SRAS:** You used a supervised machine learning approach.
        *   **Model:** Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`).
        *   **Training:** The model was trained on your `Training_Data.csv` where labels (Positive/Negative) were derived from star ratings.
        *   **Parameters:** You used `class_weight='balanced'` to help with potential class imbalance and `random_state=0` for reproducibility.

**IV. Model Evaluation (Standard Machine Learning Practice, relevant to NLP tasks)**

1.  **Train-Test Split:**
    *   **SRAS:** You split your data into training and testing sets (`sklearn.model_selection.train_test_split`) to evaluate how well your trained model generalizes to unseen data. Stratification was used.
2.  **Performance Metrics:**
    *   **SRAS:** You used standard classification metrics to evaluate your Decision Tree model:
        *   Accuracy
        *   Precision
        *   Recall
        *   F1-Score
        *   Confusion Matrix (TN, FP, FN, TP)

**V. Software Development and System Integration Techniques (Not strictly NLP, but part of your SRAS project implementation)**

1.  **API Development (Backend):**
    *   **SRAS:** Used Python with the **FastAPI** framework to create a web API that serves your trained ML model.
    *   Used **Pydantic** for data validation of API request bodies.
    *   Used **Uvicorn** as the ASGI server to run FastAPI.
2.  **Frontend Development:**
    *   **SRAS:** Used **React** (a JavaScript library) to build the user interface.
    *   Used standard web technologies: **HTML, CSS, JavaScript**.
    *   Used **Axios** for making HTTP requests from the frontend to the backend API.
3.  **Model Persistence:**
    *   **SRAS:** Used Python's **`pickle`** library to serialize (save) your trained TF-IDF vectorizer and Decision Tree model to `.pkl` files, and then deserialize (load) them in your backend API.
4.  **Data Handling (General):**
    *   **SRAS:** Used the **Pandas** library for loading and manipulating your `Training_Data.csv`.
5.  **Utility Scripting:**
    *   **SRAS:** Used Python with `os` and `shutil` modules in `update_model.py` to manage and copy model files.

**Summary List of Techniques Used in SRAS:**

*   **NLP Preprocessing:**
    *   Text Cleaning (Regex)
    *   Lowercasing
    *   Tokenization (NLTK)
    *   Stop-word Removal (NLTK)
*   **Feature Extraction:**
    *   TF-IDF (Scikit-learn `TfidfVectorizer`) with unigrams and bigrams
*   **Machine Learning for Sentiment Classification:**
    *   Supervised Learning
    *   Decision Tree Classifier (Scikit-learn `DecisionTreeClassifier`)
    *   Handling Class Imbalance (using `class_weight='balanced'`)
*   **Model Evaluation:**
    *   Train-Test Split
    *   Accuracy, Precision, Recall, F1-Score, Confusion Matrix
*   **Rule-Based Logic:**
    *   Sentiment-Rating Alignment check
*   **Backend Development:**
    *   API creation with FastAPI (Python)
    *   Data validation with Pydantic
*   **Frontend Development:**
    *   User Interface with React (JavaScript)
    *   API communication with Axios
*   **Model Persistence:**
    *   Serialization/Deserialization with Pickle
*   **Data Handling:**
    *   Data manipulation with Pandas
*   **General Python Scripting**

This list covers the main techniques and methods evident from your project's description and codebase, aligning them with common NLP and machine learning practices.