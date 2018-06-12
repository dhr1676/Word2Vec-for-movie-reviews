import re
import time
import logging

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

import nltk.data
# nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from gensim.models import Word2Vec


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())

    # 2. Loop over each sentence and append them into one list
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))

    return sentences


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    vec_feature = np.zeros((num_features,), dtype="float32")

    num_words = 0

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    # here is model.wv.index2word
    index2word_set = set(model.wv.index2word)

    # if it is in the model's vocabulary,
    # add its feature vector to the total
    for word in words:
        if word in index2word_set:
            num_words += 1
            vec_feature = np.add(vec_feature, model[word])

    # Divide the result by the number of words to get the average
    vec_feature = np.divide(vec_feature, num_words)

    return vec_feature


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    counter = 0

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1

    return reviewFeatureVecs


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


if __name__ == '__main__':
    # Read data
    train = pd.read_csv("./input_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./input_data/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("./input_data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    model = Word2Vec.load("300-features_40-min_word_10-context")
    # print(type(model.wv.syn0))
    # print(model.wv.syn0[0])
    # print(model["flower"].shape)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality

    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop words removal.
    # print("Creating average feature vectors for train reviews")
    # clean_train_reviews = []
    # for review in train["review"]:
    #     clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    #
    # trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)
    #
    # print("Creating average feature vectors for test reviews")
    # clean_test_reviews = []
    # for review in test["review"]:
    #     clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    # testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
    #
    # forest = RandomForestClassifier(n_estimators=100)
    #
    # print("Fitting a random forest to labeled training data...")
    # forest = forest.fit(trainDataVecs, train["sentiment"])
    #
    # # Test and extract results
    # result = forest.predict(testDataVecs)
    #
    # output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    #
    # output.to_csv("Word2Vec_AverageVectors.csv", index=False, quoting=3)

    start = time.time()
    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 5)
    # print(word_vectors.shape)

    # Initialize a k-means object and use it to extract centroids
    print("Running K means")
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print("Time take for K-Means clustering: %f seconds" % elapsed)

    # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # For the first 10 clusters
    # for cluster in range(0, 10):
    #     # Print the cluster number
    #     print("\nCluster %d" % cluster)
    #
    #     # Find all of the words for that cluster number, and print them out
    #     words = []
    #     for i in range(0, len(word_centroid_map.values())):
    #         if word_centroid_map.values()[i] == cluster:
    #             words.append(word_centroid_map.keys()[i])
    #     print(words)

    print("Cleaning training reviews")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    print("Cleaning test reviews")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    # ****** Create bags of centroids
    #
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("./output/BagOfCentroids.csv", index=False, quoting=3)
    print("Wrote BagOfCentroids.csv")
