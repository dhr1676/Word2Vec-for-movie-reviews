import re
# import time
# import logging

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

import nltk
# nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)

    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)

    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()

    # 4. In Python, searching a set is much faster than searching a list,
    # so convert the stop words to a set
    stops = set(stopwords.words("english"))

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]

    # 6. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


if __name__ == '__main__':
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    print("Loading train data\n")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    # Get the number of reviews based on the data column size
    num_reviews = train["review"].size

    print("Cleaning and parsing the training set movie reviews...\n")
    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review;
    # for i in range(0, int(num_reviews / 5)):
    for i in range(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        if (i + 1) % 5000 == 0:
            print("Train data Review %d of %d\n" % (i + 1, num_reviews))
        # Call our function for each one, and add the result to the list of clean reviews
        clean_train_reviews.append(review_to_words(train["review"][i]))
    print("train data clean finish!\n")

    print("Creating the bag of words...\n")
    # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    # fit_transform() does two functions:
    # First, it fits the model and learns the vocabulary;
    # second, it transforms our training data into feature vectors.
    # The input to fit_transform should be a list of strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)

    # Numpy arrays are easy to work with, so convert the result to an array
    # 这块得更新一下，toarray()这个不能用了，用toarray()会导致MemoryError
    # train_data_features = train_data_features.toarray()
    np.array(train_data_features)
    print("train_data_features shape", train_data_features.shape)

    # # Take a look at the words in the vocabulary
    # vocab = vectorizer.get_feature_names()
    # print(vocab[1:20])
    #
    # # Sum up the counts of each vocabulary word
    # dist = np.sum(train_data_features, axis=0)
    #
    # # For each, print the vocabulary word and the number of times it
    # # appears in the training set
    # for count, tag in sorted([(count, tag) for tag, count in zip(vocab, dist)], reverse=True)[1:20]:
    #     print(count, tag)

    print("Training the random forest...")
    # Initialize a Random Forest classifier with 100 trees
    forest = RandomForestClassifier(n_estimators=100, n_jobs=4)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_data_features, train["sentiment"])

    # Read the test data
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify that there are 25000 rows and 2 columns
    print("Test data shape", test.shape)

    num_test_reviews = len(test["review"])
    clean_test_reviews = []

    print("Cleaning and parsing the test set movie reviews...\n")
    for i in range(0, num_test_reviews):
        if (i+1) % 5000 == 0:
            print("Test data Review %d of %d\n" % (i+1, num_test_reviews))
        clean_review = review_to_words(test["review"][i])
        clean_test_reviews.append(clean_review)

    print("Make prediction...\n")
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.array(test_data_features)

    result_rf = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and a "sentiment" column
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result_rf})

    # Use pandas to write the comma-separated output file
    output.to_csv("Bag_of_Words_model_output.csv", index=False, quoting=3)
    print("Predict finished!\n")
