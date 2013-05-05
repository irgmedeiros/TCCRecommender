from math import sqrt

__author__ = 'Igor Medeiros'

import numpy as np
import numpy.linalg as LA

similarities_mesures = (
        'sim_distance',
        'sim_pearson',
        'sim_cosine',
        'adjusted_cosine'
        )

similarity_types = (
        'item',
        'user'
        )

class Similarity(object):
    """
    This class is responsible for handle calculations of similarities between items or users.
    Calculate the item-item and user-user similarity matrix
    """
    # TODO create a base similarity for differentiate item similarity and user similarity
    def __init__(self, model, type_sim, inverted_model=None, similarity="sim_distance"):
        super(Similarity, self).__init__()
        self.type_sim = type_sim
        if self.type_sim not in similarity_types and type_sim is not None:
            raise ValueError('similarity type %s is not recognized. valid keywords \
              are %s' % (type, list(similarity_types)))
        else:
            # similarity for item centric mode
            if type_sim == 'item':
                self.model = inverted_model
                self.inverted_model = model
                #pre calculate average user ratings for optimize adjusted cosine
                self.averageSum = self.build_item_mean(model)
            # similarity for user centric mode
            else:
                self.model = model
                self.inverted_model = inverted_model
                self.averageSum = self.build_item_mean(inverted_model)

        if similarity not in similarities_mesures and similarity is not None:
            raise ValueError('similarity %s is not recognized. valid keywords \
              are %s' % (similarity, list(similarities_mesures)))
        else:
            self.similarity = similarity

    # Calculate de similarity between p1 and p2
    def calculate(self, p1, p2):
        if "adjusted_cosine" == self.similarity:
            return self.adjusted_cosine(p1, p2)
        elif "sim_pearson" == self.similarity:
            return self.sim_pearson(p1, p2)
        elif "sim_distance" == self.similarity:
            return self.sim_distance(p1, p2)
        elif "sim_cosine" == self.sim_cosine:
            return self.sim_cosine(p1, p2)

    # Returns the itemSimilarity matrix
    def build_itemSimilarity(self, model_size=None):
        """ Build a dictionary for retrieve pre calculated similarity.
            if model_size is None considere all items in a full relation item-item
            model_size is the number of items to considererate in the model
        """
        raise NotImplementedError

    # Returns the userSimilarity matrix
    def build_userSimilarity(self):
        """ Build a dictionary for retrieve pre calculated similarity """
        raise NotImplementedError

    # Returns a distance-based similarity score for person1 and person2
    def sim_distance(self, p1, p2):
        prefs = self.model
        # Get the list of shared_items
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1

        # if they have no ratings in common, return 0
        if len(si) == 0:
            return 0

        a = np.array([prefs[p1][it] for it in si], np.float)
        b = np.array([prefs[p2][it] for it in si], np.float)

        return 1 / (1 + np.sqrt(np.sum((a - b)**2)))


    # Returns the Pearson correlation coefficient for p1 and p2
    def sim_pearson(self, p1, p2):
        prefs = self.model
        # Get the list of mutually rated items
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1

        # if they are no ratings in common, return 0
        if len(si) == 0:
            return 0

        # Sum calculations
        n = len(si)

        # Sums of all the preferences
        sum1 = sum([prefs[p1][it] for it in si])
        sum2 = sum([prefs[p2][it] for it in si])

        # Sums of the squares
        sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
        sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

        # Sum of the products
        pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

        # Calculate r (Pearson score)
        num = pSum - (sum1 * sum2 / n)
        den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
        if den == 0:
            return 0

        r = num / den

        return r

    def sim_cosine(self, p1, p2):
        prefs = self.model
        # Get the list of mutually rated items
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1

        # if they are no ratings in common, return 0
        if len(si) == 0:
            return 0


        a = np.array([prefs[p1][it] for it in si], np.float)
        b = np.array([prefs[p2][it] for it in si], np.float)

        # a function to calculate cosine similarity
        return np.inner(a, b)/(LA.norm(a)*LA.norm(b))


    def adjusted_cosine(self, p1, p2):
        '''
        >>> X = np.array([1,5,4])
        '''
        prefs = self.model
        itemPrefs = self.inverted_model

        # Get the list of mutually rated items
        si = {}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item] = 1

        # if they are less than 5 ratings in common, return 0
        # this prevents unpopular items to be too similar to another

        if len(si) < 5:
            return 0

        # Ratings vector of p1
        X = np.array([prefs[p1][it] for it in si])

        # Ratings vector of p2
        Y = np.array([prefs[p2][it] for it in si])

        # Vector with mean ratings of common itens
        mI = [self.averageSum[it] for it in si]


        # Preferences minus average rating of item
        X_mI = X - mI
        Y_mI = Y - mI

        num = sum(X_mI * Y_mI)
        den = np.sqrt(np.sum(X_mI **2)) * np.sqrt(np.sum(Y_mI **2))

        return num / den if den != 0 else 0

    def build_item_mean(self, model):
        result = {}

#        c = 0
        for key in model:
#            c += 1
#            if c % 50 == 0: print "%d / %d" % (c, len(model))
            result[key] = np.mean(model[key].values())

        return result