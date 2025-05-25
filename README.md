# Sentiment and Rating Analysis System (SRAS) for Product Reviews

## 🔍 Introduction

**SRAS** (Sentiment and Rating Analysis System) is a full-stack machine learning web application that analyzes product reviews to determine their sentiment (positive or negative) and checks if the sentiment aligns with the given user rating. It is built with a **FastAPI backend**, a modern **React.js frontend** styled with **Tailwind CSS**, and includes a custom **model training pipeline** using **scikit-learn**, **NLTK**, and **TF-IDF vectorization**.

---

## 🛠️ Technologies Used

### 🔙 Backend
- Python 3.x
- FastAPI
- scikit-learn
- pandas
- matplotlib (for visualization)
- seaborn (optional styling)
- NLTK (tokenization + stopwords)
- pickle (for model storage)
- WordCloud (for review analysis)
- tqdm (for preprocessing progress bar)

### 🔜 Frontend
- React.js (via Vite)
- Axios (for API interaction)
- Tailwind CSS (UI Styling)

---

## 🗂️ Project Structure

```
SRAS Product Reviews Project/
│
├── backend/
│   ├── api.py               # API for sentiment analysis and rating evaluation
│   ├── model.pkl            # Serialized Decision Tree model (generated after training)
│   └── vectorizer.pkl       # Serialized TF-IDF vectorizer (generated after training)
│
├── frontend/
│   └── sras-ui/
│       ├── node_modules/
│       └── src/
│           ├── App.jsx      # Main frontend component
│           ├── App.css
│           ├── index.css
│           └── main.jsx
│
├── training/
│   ├── Training_Data.csv    # Labeled dataset for training
│   ├── Training_py.py       # Core training script wrapped in a function
│   └── update_model.py      # Automates training and moves model to backend
│
├── requirements.txt         # Python backend dependencies
└── README.md                # Project documentation
```

---

## ✨ Features

1. **Sentiment Analysis**

   * Predicts sentiment of the review (Positive or Negative) using trained ML model.

2. **Rating Relevance Check**

   * Compares the predicted sentiment with the numeric rating (e.g., 1–5 stars) to determine if the rating reflects the actual sentiment.

3. **Live Review Input**

   * Users can enter reviews and ratings in the frontend to get instant feedback.

4. **Retrainable Model Pipeline** with automated packaging

   * Just run `update_model.py` with new data to retrain and update your ML pipeline.

5. **Tokenization + Stopword Removal** with NLTK

6. **React-FastAPI Integration** using Axios

7. **Modern Web UI** to input and analyze reviews

   * Decoupled frontend (React) and backend (Python) for better scalability and maintainability.

8. **Clear Separation of Concerns**

   * Decoupled frontend (React) and backend (Python) for better scalability and maintainability.

---

## 🚀 Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/aakash4456/SRAS-Product-Reviews.git
cd "SRAS Product Reviews Project"
```

---

### Step 2: Backend Setup
```bash
cd backend
pip install -r ../requirements.txt
```

Ensure `nltk` downloads are in place. Run once:
Open a Python shell and run:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

Now start the backend:
```bash
uvicorn api:app --reload
```

---

### Step 3: Train or Update the Model
```bash
cd training
python update_model.py
```

This will:
- Train the model from `Training_Data.csv`
- Generate `model.pkl` and `vectorizer.pkl`
- Move both into `/backend`

---

### Step 4: Frontend Setup
```bash
cd ../frontend/sras-ui
npm install
npm run dev
```
Visit: [http://localhost:5173](http://localhost:5173)

---

## 📦 requirements.txt

```
fastapi
uvicorn
pydantic
scikit-learn
pandas
nltk
matplotlib
seaborn
tqdm
wordcloud
```

Use:
```bash
pip install -r requirements.txt
```

---

## 🔮 Future Enhancements

* Aspect-based sentiment analysis
* Multi-class sentiment classification (positive, negative, neutral)
* Model explainability using LIME or SHAP
* Sentiment trends visualization
* Admin panel to manage reviews and retrain models

---

## 🤝 Contributing

We welcome contributions to improve SRAS!

1. Fork the Repository
2. Create a Feature Branch

```bash
git checkout -b your-feature-name
```

3. Make Your Changes
4. Commit and Push

```bash
git commit -m "Add: Your feature"
git push origin your-feature-name
```

5. Open a Pull Request 🚀

---

## 📬 Contact

* Email: [aakash4456bhu@gmail.com](mailto:aakash4456bhu@gmail.com)
* GitHub: [@aakash4456](https://github.com/aakash4456)

---