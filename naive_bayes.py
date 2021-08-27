"""
    Naive Bayes Class
        - Bernoulli Naive Bayes
        - Multinomial Naive Bayes
        - Gaussian Naive Bayes
    Arthor: Zhenhuan(Steven) Sun
"""

import numpy as np

class BernoulliNB:
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
        conditional_prob_denominator = np.expand_dims(np.array([len(x) + 2 * self.K for x in X_separated_by_class]), axis=1)
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

class MultinomialNB:
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
        # assume we have seen each feature at least once to avoid divide by zero error
        conditional_prob_numerator = np.array([np.array(x).sum(axis=0) + self.K for x in X_separated_by_class])

        # compute the conditional probability's denominator for different class
        conditional_prob_denominator = np.expand_dims(conditional_prob_numerator.sum(axis=1), axis=1)

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

class GaussianNB:
    def __init__(self, k=1.0):
        # Laplace Smoothing Factor
        self.K = k

    def fit(self, X, y):
        # separate the training set by classes
        X_separated_by_class = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

        # count the number of different classes
        self.n_classes = len(np.unique(y))

        # compute prior probability
        self.prior_prob = np.array([len(x) / X.shape[0] for x in X_separated_by_class])

        # compute mean vector for each class
        self.mean_vector = np.array([np.array(x).sum(axis=0) / len(x) for x in X_separated_by_class])

        # compute covariance matrix for each class
        covariance_diagonal_matrices = []
        for c, x in enumerate(X_separated_by_class):
            mean_square_difference = 0
            for x_i in x:
                # compute the covariance matrix for each examples (slow as hell -> abandoned)
                # mean_difference = np.expand_dims((x_i - self.mean_vector[c]), axis=1)
                # mean_square_difference += mean_difference.dot(mean_difference.T) 
                # compute the diagnal entries of covariance matrix for each examples (much faster than above method)
                mean_difference = x_i - self.mean_vector[c]
                mean_square_difference += mean_difference ** 2
            # convert the list of diagonal entries back to covariance diagonal matrix
            # here we assumed that the mean square difference between each feature and its mean is at least 1 to make sure that 
            # there is no zero variance in the covariance matrix and thus we won't encounter divide by zero error in the future
            covariance_diagonal_matrix = ((mean_square_difference + self.K) / len(x)) * np.identity(X.shape[1])
            covariance_diagonal_matrices.append(covariance_diagonal_matrix)
        self.covariance_diagonal_matrices = np.asarray(covariance_diagonal_matrices)

        return self

    def log_gaussian_distribution(self, x, mean, variance):

        log_multiplier = -np.log(np.sqrt((2 * np.pi) * variance))
        log_exponent = -(x - mean)**2 / (2 * variance)

        return sum(log_multiplier + log_exponent)

    def predict(self, X):
        variances = []
        for matrix in self.covariance_diagonal_matrices:
            variance = matrix.diagonal()
            variances.append(variance)
        variances = np.array(variances)
        
        # list that stores all test data's posterior probability
        posterior_prob_collection = []
        for x in X:
            conditional_prob = []
            for mean, variance in zip(self.mean_vector, variances):
                # compute conditional probability for each class
                conditional_prob.append(self.log_gaussian_distribution(x, mean, variance))
            # compute posterior probability
            posterior_prob = np.array(conditional_prob) + np.log(self.prior_prob)
            posterior_prob_collection.append(posterior_prob)
        posterior_prob_collection = np.array(posterior_prob_collection)
        
        return np.argmax(posterior_prob_collection, axis=1)