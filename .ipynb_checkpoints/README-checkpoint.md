# Sentiment and Rating Analysis System(SRAS) for Product Reviews

## Introduction

The SRAS for Product Reviews is a Python-based application designed to analyze product reviews and predict their sentiment as positive or negative. The project leverages natural language processing (NLP) techniques, including TF-IDF vectorization, and employs a Decision Tree classifier for prediction. The project also features a GUI built with Tkinter for user interaction, allowing users to analyze reviews and evaluate rating relevance.

## Prerequisites

Before you begin, ensure the following are installed on your machine:

* Python (v3.7 or higher)
* pip (Python package installer)

## Setup Instructions

1. Clone the Repository

```bash
git clone https://github.com/aakash4456/SRAS.git
cd SRAS
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Download NLTK Resources
Open a Python shell and run:
```bash
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

4. Run the Application
To launch the GUI application, execute:
```bash
python App.py
```

## Project Structure

```bash
src/
.
├── Training_py.py            # Core script for training and saving the model
├── app.py                 # GUI application for analyzing reviews
├── Training_Data.csv      # Dataset containing reviews
├── requirements.txt       # Python dependencies
├── vectorizer.pkl         # Saved TF-IDF vectorizer
├── model.pkl              # Trained Decision Tree model
└── README.md              # Project documentation
```

## Features

1. Sentiment Analysis:
* Classifies reviews as positive or negative based on their content.

1. Rating Relevance Check:
* Compares predicted sentiment with user-provided ratings to determine alignment.
* 
1. Word Cloud Visualization:
* Generates a word cloud for positive reviews as an optional step.
* 
1. GUI:
* Provides a user-friendly interface for entering reviews and ratings, displaying results instantly.

## Key Technologies

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Decision Tree Classifier
* Tkinter

## Contributing

We welcome contributions to this project! Follow these steps:

1. Fork the Repository: Create your own fork of the repository.
2. Create a Branch:
```bash
git checkout -b feature-branch-name
```
3. Make Your Changes: Implement your feature or fix.
4. Commit Your Changes:
```bash
git commit -m "Describe your changes"
```
5. Push to Your Branch:
```bash
git push origin feature-branch-name
```
6. Create a Pull Request: Submit your branch for review.

## Contact

For questions, suggestions, or feedback, feel free to reach out:



- Email: [aakash4456bhu@gmail.com](aakash4456bhu@gmail.com)
- GitHub: [GitHub Profile](https://github.com/aakash4456)