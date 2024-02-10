import csv
import math
import random
import time
import pickle as pk
import pandas as pd
from sklearn.decomposition import PCA
from hazm import *
import numpy as np
import sklearn


def train_test_split(data, tags):
    # Splits data into training set and testing set
    train_data = data[2000:18000]
    test_data = data[:2000] + data[18000:]
    train_tags = tags[2000:18000]
    test_tags = tags[:2000] + tags[18000:]
    return train_data, train_tags, test_data, test_tags


def calculate_accuracy(x, y, test, test_y, k, function):
    # uses f-score to calculate accuracy
    tp, fp, fn, tn = [], [], [], []
    print('gonna enter knn')
    predictions = function(x, y, test, k)
    print('knn finished')
    for i in range(len(predictions)):
        print('supposed to be: ', test_y[i], ' is actually: ', predictions[i])
        tp.append(np.sum(predictions[i] == test_y[i] and predictions[i] == 1))
        fp.append(np.sum(predictions[i] == 1 and test_y[i] == 0))
        fn.append(np.sum(predictions[i] == 0 and test_y[i] == 1))
        tn.append(np.sum(predictions[i] == test_y[i] and predictions[i] == 0))

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)

    f1 = (2 * tp) / (2 * tp + fn + fp)

    return f1


def knn_classifier(X_train, y_train, X_test, k):
    # implementation of knn classifier

    distances = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(X_test.shape[0]):
        for j in range(X_train.shape[0]):
            distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])
    nearest_neighbors = []
    for i in range(X_test.shape[0]):
        k_nearest_indices = np.argpartition(distances[i], k)[:k]
        k_nearest_classes = y_train[k_nearest_indices]
        nearest_neighbors.append(k_nearest_classes)

    predictions = []
    for neighbors in nearest_neighbors:
        class_counts = np.unique(neighbors, return_counts=True)
        predicted_class = class_counts[0][np.argmax(class_counts[1])]
        predictions.append(predicted_class)
    return predictions


def term_document_matrix(documents, vocabulary):
    # Uses a list to initialize the term document matrix

    dtm = []
    for document in documents:
        document_vector = np.zeros(len(vocabulary), dtype='uint8').tolist()
        for word in document:
            index = vocabulary.get(word)
            if index is not None:
                document_vector[index] += 1

        dtm.append(document_vector)

    return dtm


# Uses hazm normalizer and Lemmatizer to clean data
normalizer = Normalizer()
lemmatizer = Lemmatizer()
tokenizer = WordTokenizer(join_verb_parts=True)
tagger = POSTagger(model='pos_tagger.model', universal_tag=True)
documents = []
categories = []
vocab = {}
vocab_index = 0
with open("nlp_test.csv", 'r', encoding='utf-8-sig') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            tokens = tokenizer.tokenize(normalizer.normalize(row[1]))
            count = 0
            for token in tokens:
                tokens[count] = lemmatizer.lemmatize(token)
                count += 1
            tags = tagger.tag(tokens=tokens)
            tags = [tag for tag in tags if
                    tag[1] not in ('ADP', 'CCONJ', 'PUNCT', 'PRON', 'SCONJ', 'DET', 'VERB', 'NUM')]
            row[1] = [tag[0] for tag in tags]
            line_count += 1
            documents.append(row[1])
            if row[2] == 'Sport':
                categories.append(1)
            else:
                categories.append(0)
            for word in row[1]:
                if word not in vocab:
                    vocab[word] = vocab_index
                    vocab_index += 1
            if line_count >= 2000:
                break

# Load the necessary data and models
pca_fit = pk.load(open("pca.pkl", 'rb'))
pca_reload = pk.load(open("pcatransform.pkl", 'rb'))
pca_categories = pk.load(open("categories.pkl", "rb"))
vocab = pk.load(open("vocab.pkl", "rb"))

# Calculate the term document matrix for test data
dtm_test = term_document_matrix(documents, vocab)

# Transform test data using PCA
x_test = pca_fit.transform(dtm_test)

# Calculate and print the accuracy using KNN
accuracy = calculate_accuracy(pca_reload, pca_categories, x_test, categories, k=13, function=knn_classifier)
print(f"Accuracy using KNN with k=13: {accuracy}")


