"""FAQ chatbot engine: NLTK preprocessing + TF-IDF + cosine similarity."""

import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tokenizer and stemmer need no downloads
_tokenizer = RegexpTokenizer(r"[a-z0-9]+")
_stemmer = PorterStemmer()


def _load_stopwords():
    """Use NLTK stopwords if available, otherwise fall back to scikit-learn's list."""
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = set(stopwords.words("english"))
    except Exception:
        words = set(ENGLISH_STOP_WORDS)
    # Keep negations, they change the meaning of a question
    return words - {"not", "no", "cannot"}


STOP_WORDS = _load_stopwords()

SMALL_TALK = {
    "greeting": (
        {"hi", "hello", "hey", "hii", "hola", "namaste"},
        "Hello! I'm the ShopEasy assistant. Ask me about orders, delivery, returns, payments and more.",
    ),
    "thanks": (
        {"thanks", "thank", "thankyou", "thx"},
        "You're welcome! Let me know if you need anything else.",
    ),
    "bye": (
        {"bye", "goodbye", "exit", "quit"},
        "Goodbye! Have a great day.",
    ),
}

FALLBACK = (
    "Sorry, I couldn't find a good answer to that. Try rephrasing your question, "
    "or contact our support team at support@shopeasy.example."
)


def preprocess(text: str) -> str:
    """Lowercase, tokenize, remove stopwords and stem."""
    tokens = _tokenizer.tokenize(str(text).lower())
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(_stemmer.stem(t) for t in tokens)


class FAQChatbot:
    def __init__(self, csv_path: str, threshold: float = 0.25):
        self.threshold = threshold
        self.df = pd.read_csv(csv_path).dropna()
        self.questions = self.df["question"].tolist()
        self.answers = self.df["answer"].tolist()

        processed = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(processed)

    def small_talk(self, text: str):
        words = set(_tokenizer.tokenize(text.lower()))
        if len(words) > 3:
            return None
        for keywords, reply in SMALL_TALK.values():
            if words & keywords:
                return reply
        return None

    def get_answer(self, user_question: str) -> dict:
        """Return the best matching FAQ answer with its similarity score."""
        reply = self.small_talk(user_question)
        if reply:
            return {"answer": reply, "score": 1.0, "matched": None}

        cleaned = preprocess(user_question)
        if not cleaned:
            return {"answer": FALLBACK, "score": 0.0, "matched": None}

        vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = int(scores.argmax())
        score = float(scores[best])

        if score < self.threshold:
            return {"answer": FALLBACK, "score": score, "matched": None}
        return {"answer": self.answers[best], "score": score, "matched": self.questions[best]}
