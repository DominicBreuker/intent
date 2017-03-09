import pickle
import spacy
import numpy as np
from keras.models import load_model


class FakeEmbedder(object):
    def __init__(self, max_length=100):
        self.max_length = max_length

    def embed(self, text):
        return np.random.random((self.max_length, 300))


class Embedder(object):
    def __init__(self, max_length=100):
        self.max_length = max_length
        print("Loading spacy...")
        self.nlp = spacy.load('de')
        print("...done")

    def embed(self, text):
        return self._pad(self._vectors(text))

    def shape(self):
        return (self.max_length, 300)

    def _vectors(self, text):
        doc = self.nlp(text)
        vectors = []
        for token in doc:
            vectors.append(token.vector)
        return vectors

    def _pad(self, vectors):
        vector_dim = len(vectors[0])
        sequence = np.zeros((self.max_length, vector_dim))
        for i, vector in enumerate(vectors):
            if i == self.max_length:
                break
            sequence[i] = vector
        return sequence


class Predictor(object):
    def __init__(self, embedder, model_path, intent_names_path):
        self.model = load_model(model_path)
        self.intent_names = pickle.load(open(intent_names_path, "rb"))
        self.embedder = embedder

    def predict(self, text):
        proba = self._predict_proba(text)
        return self.intent_names[np.argmax(proba)]

    def _predict_proba(self, text):
        embedding = self.embedder.embed(text)
        return self.model.predict(np.array([embedding]))
