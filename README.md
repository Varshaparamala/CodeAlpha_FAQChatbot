# CodeAlpha_FAQChatbot

An FAQ chatbot for a fictional online store, built for the CodeAlpha AI Internship (Task 2).

## How it works
1. **Collect FAQs:** 20 question-answer pairs are stored in `faqs.csv`.
2. **Preprocess:** each question is lowercased, tokenized, stripped of stop words and stemmed using NLTK.
3. **Vectorize:** questions are converted to TF-IDF vectors (unigrams + bigrams).
4. **Match:** the user's question goes through the same preprocessing, and cosine similarity finds the closest FAQ.
5. **Reply:** the best answer is shown. If the similarity is below the threshold, the bot says it didn't understand.

## Features
- Chat UI built with Streamlit
- Handles greetings, thanks and goodbye
- Shows the matched FAQ and a confidence score
- Sample questions in the sidebar
- Easy to customize: just edit `faqs.csv`

## Tech Stack
Python, NLTK, scikit-learn, pandas, Streamlit

## Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Use Your Own FAQs
Replace the rows in `faqs.csv` (columns: `question`, `answer`) and restart the app.

## Screenshots
_Add screenshots here._

## Author
Varshaparamala | CodeAlpha AI Intern
