from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi.middleware.cors import CORSMiddleware


# Load model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Input schema
class ReviewRequest(BaseModel):
    review: str
    rating: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
def analyze(data: ReviewRequest):
    review = re.sub(r"[^\w\s]", "", data.review)
    tokens = [w.lower() for w in word_tokenize(review) if w.lower() not in stopwords.words('english')]
    vector = vectorizer.transform([' '.join(tokens)]).toarray()
    pred = model.predict(vector)[0]
    sentiment = "Positive" if pred == 1 else "Negative"
    relevance = "Aligned" if (sentiment == "Positive" and data.rating >= 4) or (sentiment == "Negative" and data.rating <= 2) else "Not Aligned"
    return {"sentiment": sentiment, "relevance": relevance}