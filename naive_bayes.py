import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class BernoulliNB():
    def __init__(self, k=1.0, binarize=0.0):
        # Laplace Smoothing Factor
        self.K = k

        # the degree of binarization
        self.binarize = binarize

    def fit(self, X, y):
        # binarize X
        # since we assume data is bernoulli distributed we need to make sure
        # that data consist of binary values
        X = self._binarize(X)

        # separate training data by classes(different target)
        X_separated_by_class = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

        # number of different class
        self.n_classes = len(np.unique(y))

        # count the number of examples and number of features in X
        self.n_examples, self.n_features = X.shape

        # count the number of examples that belong to class k (0 or 1 in spam classification)
        prior_numerator = np.array([len(x) for x in X_separated_by_class])

        # compute the prior probability (P(y))
        self.prior_prob = prior_numerator / self.n_examples

        # compute the log prior probability (log(P(y))) for prediction
        self.log_prior_prob = np.log(self.prior_prob)

        # compute the conditional probability
        # with laplace smoothing we assume we have seen each feature at least self.K times
        conditional_prob_numerator = np.array([np.array(x).sum(axis=0) + self.K for x in X_separated_by_class])
        conditional_prob_denominator = np.expand_dims(np.array([len(x) + self.n_classes * self.K for x in X_separated_by_class]), axis=1)
        self.conditional_prob = conditional_prob_numerator / conditional_prob_denominator

        return self

    def predict(self, X):
        # binarize X
        X = self._binarize(X)

        # compute log posterior probability log(P(y|X))
        posterior_prob_numerator = np.array([(x * np.log(self.conditional_prob) + 
                                   np.abs(1 - x) * np.log(1 - self.conditional_prob)).sum(axis=1) + 
                                   self.log_prior_prob for x in X])
        posterior_prob_denominator = np.expand_dims(np.array([(x * np.log(self.conditional_prob) + 
                                    np.abs(1 - x) * np.log(1 - self.conditional_prob)).sum(axis=1) +
                                    self.log_prior_prob for x in X]).sum(axis=1), axis=1)
                                    
        posterior_prob = posterior_prob_numerator - posterior_prob_denominator

        # alternative solution
        # since posterior_prob_denominator is a constant thus we don't bother compute the denominator
        # compute the numerator is sufficient enough to make prediction and also it makes algorithm runs faster
        #return np.argmax(posterior_prob_numerator, axis=1)

        return np.argmax(posterior_prob, axis=1)

    def _binarize(self, X):
        # convert the values in X to binary values (0 or 1)
        return np.where(X > self.binarize, 1, 0)

class MultinomialNB():
    def __init__(self, k=1.0):
        # Laplace Smoothing Factor
        self.K = k

    def fit(self, X, y):
        # separate the training data by class
        X_separated_by_class = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        
        # number of different class
        self.n_classes = len(np.unique(y))

        # count the number of examples that belong to different classes
        prior_numerator = [len(x) for x in X_separated_by_class]

        # count the total number of examples in the training set
        prior_denominator = X.shape[0]

        # compute prior probability
        self.prior_prob = np.array(prior_numerator) / prior_denominator

        # compute log prior probability for prediction
        self.log_prior_prob = np.log(self.prior_prob)

        # compute the conditional probability's numerator for different class (with laplace smoothing)
        conditional_prob_numerator = np.array([np.array(x).sum(axis=0) + self.K for x in X_separated_by_class])

        # compute the conditional probability's denominator for different class (with laplace smoothing)
        conditional_prob_denominator = np.expand_dims(conditional_prob_numerator.sum(axis=1) + self.n_classes * self.K, axis=1)

        # compute the conditional probability for each feature and for each different classes
        self.conditional_prob = conditional_prob_numerator / conditional_prob_denominator

        return self

    def predict(self, X):
        # compute the log conditional probability for each examples and for each different classes
        log_conditional_prob = np.array([(x * np.log(self.conditional_prob)).sum(axis=1) for x in X])

        # compute the posterior probability
        posterior_pronb = log_conditional_prob + self.log_prior_prob

        # make prediction
        return np.argmax(posterior_pronb, axis=1)

# read csv file
df = pd.read_csv("./Data/emails.csv")

# separate X and y from data frame
X = np.array(df.iloc[:, 1:3001])
y = df.iloc[:, 3001].values

# split data set to training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# bnb = MultinomialNB()
# bnb.fit(X_train, y_train)
# print(bnb.conditional_prob.shape)
#y_predict = bnb.predict(X_test)