"""
    1. Spam Emails Classification Problem using
        - Bernoulli Naive Bayes
        - Multinomial Naive Bayes
        - Gaussian Naive Bayes
    2. Comparison between my implementation of Naive Bayes and scikit-learn
       implementation of Naive Bayes
    Arthor: Zhenhuan(Steven) Sun
"""

import pandas as pd
import numpy as np
from naive_bayes import BernoulliNB as myBernoulliNB
from naive_bayes import MultinomialNB as myMultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

# read csv file
df = pd.read_csv("./Data/emails.csv")

# separate X and y from data frame
X = np.array(df.iloc[:, 1:3001])
y = df.iloc[:, 3001].values

# split data set to training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# scikit-learn bernoulli naive bayes classifier
bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_predict_bnb = bnb.predict(X_test)
print("Accuracy Score for Bernoulli Naive Bayes (scikit-learn implementation): ", 
        accuracy_score(y_predict_bnb, y_test))
# my implementation of bernoulli naive bayes classifier
my_bnb = myBernoulliNB()
my_bnb.fit(X_train, y_train)
y_predict_my_bnb = my_bnb.predict(X_test)
print("Accuracy Score for Bernoulli Naive Bayes (my implementation): ", 
        accuracy_score(y_predict_my_bnb, y_test))
print("\n")

# scikit-learn multinomial naive bayes classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_predict_mnb = mnb.predict(X_test)
print("Accuracy Score for Multinomial Naive Bayes (scikit-learn implementation): ", 
        accuracy_score(y_predict_mnb, y_test))
# my implementation of bernoulli naive bayes classifier
my_mnb = myMultinomialNB()
my_mnb.fit(X_train, y_train)
y_predict_my_mnb = my_mnb.predict(X_test)
print("Accuracy Score for Multinomial Naive Bayes (my implementation): ", 
        accuracy_score(y_predict_my_mnb, y_test))
print("\n")

# scikit-learn gaussian naive bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_predict_gnb = gnb.predict(X_test)
print("Accuracy Score for Gaussian Naive Bayes (scikit-learn implementation): ", 
        accuracy_score(y_predict_gnb, y_test))
print("\n")