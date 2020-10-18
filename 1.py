import tensorflow
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import json

with open("intents.json") as file:
    data = json.load(file)["intents"]

print(data)
