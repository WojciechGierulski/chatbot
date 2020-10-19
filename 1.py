import tensorflow
import random
import nltk
nltk.download("punkt")
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import json
import pickle


try:
    with open("data.pickle", "rb") as file:
        words, labels, training, output = pickle.load(file)

except:
    with open("intents.json") as file:
        intents = json.load(file)["intents"]

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in intents:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    words = [stemmer.stem(word.lower()) for word in words if word != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output= []

    out_empty = [0 for _ in range(len(labels))]

    for i, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc if doc != "?"]
        for word in words:
            if word in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[i])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=5000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

