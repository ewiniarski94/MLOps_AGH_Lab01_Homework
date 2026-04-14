import joblib
from sentence_transformers import SentenceTransformer

SENTIMENT_TO_TEXT = {0: "negative", 1: "neutral", 2: "positive"}


class SentimentModel:
    def __init__(self, transformer_path: str, classifier_path: str):
        self.transformer = SentenceTransformer(transformer_path)
        self.classifier = joblib.load(classifier_path)

    def predict(self, text: str) -> str:
        embedding = self.transformer.encode([text])
        prediction = self.classifier.predict(embedding)[0]
        return SENTIMENT_TO_TEXT.get(prediction, "invalid")


def load_sentiment_model():
    return SentimentModel(
        transformer_path="sentence_transformer.model",
        classifier_path="classifier.joblib",
    )
