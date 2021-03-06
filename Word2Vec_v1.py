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
        if counter % 5000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1

    return reviewFeatureVecs


if __name__ == '__main__':
    # Read data
    train = pd.read_csv("./input_data/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("./input_data/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("./input_data/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    # print("Read %d labeled train reviews, %d labeled test reviews, "
    #       "and %d unlabeled reviews\n" % (train["review"].size,
    #                                       test["review"].size,
    #                                       unlabeled_train["review"].size))

    # load the punctuation tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    sentences = []
    print("Parsing sentences from training set")
    for review in train["review"]:
        # The difference between the "+=" and append()
        # If you are appending a list of lists to another list of lists,
        # append() will only append the first list,
        # you will need to use "+=" in order to join all of the lists at once
        sentences += review_to_sentences(review, tokenizer)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    # Check how many sentences we have in total
    # should be around 850,000+
    print("how many sentences we have %d\n " % len(sentences))

    print(sentences[0])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words

    # Initialize and train the model
    print("Training model...")
    model = Word2Vec(sentences,
                     workers=num_workers,
                     size=num_features,
                     min_count=min_word_count,
                     window=context,
                     sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300-features_40-min_word_10-context"
    model.save(model_name)

    # Test the model
    # model = Word2Vec.load("300-features_40-min_word_10-context")
    # model.doesnt_match("man woman child kitchen".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.most_similar("man")
    # model.most_similar("queen")
    # model.most_similar("awful")

    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop words removal.
    print("Creating average feature vectors for train reviews")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, remove_stopwords=True))

    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    print("Creating average feature vectors for test reviews")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    forest = RandomForestClassifier(n_estimators=100)

    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(trainDataVecs, train["sentiment"])

    # Test and extract results
    result = forest.predict(testDataVecs)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    output.to_csv("./output/Word2Vec_AverageVectors.csv", index=False, quoting=3)

    start = time.time()
    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 5)
    # print(word_vectors.shape)

    # Initialize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    end = time.time()
    elapsed = end - start
    print("Time take for K-Means clustering: %f seconds" % elapsed)

    # # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    # word_centroid_map = dict(zip(model.wv.index2word, idx))
    # # For the first 10 clusters
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

