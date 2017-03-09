import json
import numpy as np

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model, load_model

from nlp import Embedder

import pickle

###### PREPROCESSING ######

embedder = Embedder()


def load_data(data_dir, limit=0):
    examples = []
    intent_names = {}
    with open(data_dir + '/intents.json') as data_file:
        data = json.load(data_file)
    for i, intent in enumerate(data):
        intent_names[i] = intent["name"]
        for message in intent["messages"]:
            examples.append((message, i))
    np.random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    messages, intents = zip(*examples)
    return examples, intent_names


def dummy_encode(array, num_classes=None):
    array = np.array(array)
    if num_classes is None:
        num_classes = max(array) + 1
    result = np.zeros((len(array), num_classes))
    result[np.arange(len(array)), array] = 1
    return result


def create_dataset(type):
    if type == 'train':
        data_dir = 'data/train'
    elif type == 'dev':
        data_dir = 'data/dev'
    else:
        assert(False), "Type must be train or dev"
    examples, intent_names = load_data(data_dir)
    X = []
    y = []
    for example in examples:
        message, intent = example[0], example[1]
        X.append(embedder.embed(message))
        y.append(intent)
    return np.array(X), dummy_encode(np.array(y)), intent_names

X_train, y_train, intent_names_train = create_dataset('train')
X_dev, y_dev, intent_names_dev = create_dataset('dev')


###### MODEL TRAINING ######

X, y, intent_names = X_train, y_train, intent_names_train


def build_model(input_shape):
    sequences = Input(shape=input_shape)
    x = Conv1D(16, 5, activation='relu')(sequences)
    x = MaxPooling1D(2)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    preds = Dense(len(intent_names), activation='softmax')(x)
    model = Model(input=sequences, output=preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

model = build_model((X.shape[1], X.shape[2]))
history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), nb_epoch=20, batch_size=128)


def save_model(model):
    model.save('output/mymodel')

save_model(model)
pickle.dump(intent_names, open("output/intent_names.p", "wb"))


###### MODEL USAGE ######


loaded_model = load_model('output/mymodel')
loaded_intent_names = pickle.load(open("output/intent_names.p", "rb"))


def predict(model, text):
    embedding = embedder.embed(text)
    return model.predict(np.array([embedding]))


def translate(prediction, intent_names):
    return intent_names[np.argmax(prediction)]


def suggest(text):
    prediction = predict(loaded_model, text)
    return translate(prediction, loaded_intent_names)
