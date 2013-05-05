import math
import numpy as np
from similarities.classes import Similarity


class UserBasedRecommender(object):
    """
    User Based Collaborative Filtering Recommender.


    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The User Similarity instance that will be used to
        score the users that are the most similar to the user.

    neighborhood_strategy: The user neighborhood strategy that you
         can choose for selecting the most similar users to find
         the items to recommend.
         default = NearestNeighborsStrategy

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The User Similarity instance that will be used to
        score the users that are the most similar to the user.

    `neighborhood_strategy`: The user neighborhood strategy that you
         can choose for selecting the most similar users to find
         the items to recommend.
         default = NearestNeighborsStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Examples
    -----------
    >>> movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
     'Snakes on a Plane': 3.5, \
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
     'The Night Listener': 3.0}, \
     'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
     'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
     'The Night Listener': 4.5, 'Superman Returns': 4.0, \
     'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'The Night Listener': 3.0, 'Superman Returns': 5.0, \
     'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}}
    >>> model = movies
    >>> recsys = UserBasedRecommender(model, similarity="sim_distance", neighborhood_size=None)
    >>> #Return the recommendations for the given user.
    >>> recsys.getRecommendations('Leopoldo Pires')
    ['Just My Luck', 'You, Me and Dupree']
    >>> #Return estimated preference for the given item.
    >>> recsys.estimate_preference('Sheldom', 'Superman Returns')
    4.28587167021193
    >>> #Return the 2 explanations for the given recommendation.
    >>> #recsys.recommended_because('Leopoldo Pires', 'Just My Luck',2) # expected: ['Lorena Abreu', 'Marcel Caraciolo']

    Notes
    -----------
    This UserBasedRecommender does not yet provide
    suppot for rescorer functions.

    References
    -----------
    User-based collaborative filtering recommendation algorithms by

    """

    def __init__(self, model, similarity="adjusted_cosine", neighborhood_size=None, build_userSim=False, with_preference=False):
        self.model = model
        self.itemPrefs = self._transformPrefs()
        self._similarity = Similarity(self.model, 'user', self.itemPrefs, similarity)
        self.with_preference = with_preference
        if neighborhood_size is None:
            self.neighborhood_size = None
        else:
            self.neighborhood_size = neighborhood_size
        if build_userSim:
            #TODO put a feedback telling to setup a userSimilarity before call recommendations
            self.userSimilarity = False
            self.userSimilarity = self.calculateUserSimilarities()
        else:
            self.userSimilarity = None

    def getRecommendations(self, person, how_many=None):
        #TODO usar predição no estilo do tapestry
        prefs = self.model
        simf = self._similarity

        totals = {}
        simSums = {}

        person_ratings = prefs[person].values()
        mean_person = sum(person_ratings) / float(len(person_ratings))

        # all neighbors user-user
        neighborhood = self.most_similar_users(person, self.neighborhood_size)
        for _, other in neighborhood:
            # don't compare me to myself
            if other == person: continue # no need to check
            sim = self.calculate(person, other)

            other_ratings = prefs[other].values()
            mean_other = sum(other_ratings) / float(len(other_ratings))

            for item in prefs[other]:

                # only score movies I haven't seen yet
                if item not in prefs[person]: # or prefs[person][item] == 0:
                    # Similarity * Normalized Score
                    totals.setdefault(item, 0)
                    totals[item] += (prefs[other][item] - mean_other) * sim
                    # Sum of similarities
                    simSums.setdefault(item, 0)
                    simSums[item] += abs(sim)

        # Create the normalized list
        rankings = [(mean_person + (total / simSums[item]), item) for item, total in totals.items() if simSums != 0]

        # Return the sorted list
        rankings.sort()
        rankings.reverse()
        if how_many is not None:
            rankings =  rankings[0:how_many]

        if not self.with_preference:
            rankings = [item for score, item in rankings]

        return rankings

    def estimate_preference(self, user, item, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id:  int or string
            ID of item for which wants to find the estimated preference.

        Returns
        -------
        :rtype : dict
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        '''

        totals = {}
        simSums = {}
        prefs = self.model
        person = user

        neighbors = self.most_similar_users(person, self.neighborhood_size)

        for sim, other in neighbors:
            # don't compare me to myself
            if other == person:
                continue
#            if self.userSimilarity:
#                sim = self.userSimilarity[user][other]
#            else:
#                sim = self._similarity.calculate(person, other)

            # ignore scores of zero or lower
            if sim <= 0:
                continue

            # Only scores if the neighbor rated 'item'
            if item in prefs[other]:
                # Similarity * Score
                totals.setdefault(item, 0)
                totals[item] += prefs[other][item] * sim
                # Sum of similarities
                simSums.setdefault(item, 0)
                simSums[item] += sim

        result = totals[item] / float(simSums[item]) if simSums[item] != 0 else None

        return result

    def topMatches(self, person, n=5, similarity="sim_pearson"):
        """
        n determines the number of people to considere
        If n is none consider every person
        """
        prefs = self.model
        sim = self._similarity
#        scores = [(sim.calculate(person, other), other) for other in prefs if other != person]

        if self.userSimilarity:
            scores = self.userSimilarity[person]
        else:
            scores = [(sim.calculate(person, other), other) for other in prefs if other != person]
        scores.sort()
        scores.reverse()
        return scores[0:n] if n is not None else scores

    def _transformPrefs(self):
        prefs = self.model
        result = {}
        for person in prefs:
            for item in prefs[person]:
                result.setdefault(item, {})
                # Flip item and person
                result[item][person] = prefs[person][item]
        return result

    def calculateUserSimilarities(self):
        prefs = self.model
        # Create a dictionary of users showing which other users they
        # are most similar to.
        result = {}

        # Computes similarities for every user. Expensive computation
        c = 0
        for item in prefs:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0: print "%d / %d" % (c, len(prefs))
            # Find the most similar useer to this one
            scores = self.topMatches(item, None, self._similarity.similarity)
            result[item] = scores

        return result

    def most_similar_users(self, user_id, how_many=None):
        '''
        Return the most similar users to the given user, ordered
        from most similar to least

        Parameters
        -----------
        user_id:  int or string
            ID of user for which to find most similar other users

        how_many: int
            Desired number of most similar users to find (default=None ALL)
        '''
        return self.topMatches(user_id,how_many,self._similarity.similarity)

    def recommended_because(self, user_id, item_id, how_many=None, **params):
        '''
        Returns the users that were most influential in recommending a
        given item to a given user. In most implementations, this
        method will return users that prefers the recommended item and that
        are similar to the given user.

        Parameters
        -----------
        user_id : int or string
            ID of the user who was recommended the item

        item_id: int or string
            ID of item that was recommended

        how_many: int
            Maximum number of items to return (default=None ALL)

        Returns
        ----------
        The list of items ordered from most influential in
        recommended the given item to least
        '''
        raise NotImplementedError

    def calculate(self, person, other):

        sim = 0
        #TODO convert userSimilarity to dict for better performance over list
        if self.userSimilarity:
            for simi, neighbor in self.userSimilarity[person]:
                if neighbor ==  other:
                    sim = simi
                    break
        else:
            sim = self._similarity.calculate(person, other)
        return sim


class ItemBasedRecommender(object):
    """
    Item Based Collaborative Filtering Recommender.


    Parameters
    -----------
    data_model: The data model instance that will be data source
         for the recommender.

    similarity: The Item Similarity instance that will be used to
        score the items that will be recommended.

    items_selection_strategy: The item candidates strategy that you
     can choose for selecting the possible items to recommend.
     default = ItemsNeighborhoodStrategy

    capper: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    with_preference: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Attributes
    -----------
    `model`: The data model instance that will be data source
         for the recommender.

    `similarity`: The Item Similarity instance that will be used to
        score the items that will be recommended.

    `items_selection_strategy`: The item candidates strategy that you
         can choose for selecting the possible items to recommend.
         default = ItemsNeighborhoodStrategy

    `capper`: bool (default=True)
        Cap the preferences with maximum and minimum preferences
        in the model.
    `with_preference`: bool (default=False)
        Return the recommendations with the estimated preferences if True.

    Examples
    -----------
    >>> movies = {'Marcel Caraciolo': {'Lady in the Water': 2.5, \
     'Snakes on a Plane': 3.5, \
     'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, \
     'The Night Listener': 3.0}, \
     'Paola Pow': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, \
     'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 3.5}, \
    'Leopoldo Pires': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, \
     'Superman Returns': 3.5, 'The Night Listener': 4.0}, \
    'Lorena Abreu': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, \
     'The Night Listener': 4.5, 'Superman Returns': 4.0, \
     'You, Me and Dupree': 2.5}, \
    'Steve Gates': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0, \
     'You, Me and Dupree': 2.0}, \
    'Sheldom': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, \
     'The Night Listener': 3.0, 'Superman Returns': 5.0, \
     'You, Me and Dupree': 3.5}, \
    'Penny Frewman': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0, \
    'Superman Returns':4.0}, \
    'Maria Gabriela': {}}
    >>> model = movies
    >>> recsys = ItemBasedRecommender(model, similarity="sim_pearson", model_size=10)
    >>> #Return the recommendations for the given user.
    >>> recsys.getRecommendedItems('Leopoldo Pires')
    ['Just My Luck', 'You, Me and Dupree']
    >>> #Return estimated preference for the given item
    >>> recsys.estimate_preference('Sheldom', 'Superman Returns')
    4.174661272381852
    >>> #Return the 2 explanations for the given recommendation.
    >>> #recsys.recommended_because('Leopoldo Pires', 'Just My Luck',2)
    >>> #['The Night Listener', 'Superman Returns']

    Notes
    -----------
    This ItemBasedRecommender does not yet provide
    suppot for rescorer functions.

    References
    -----------
    Item-based collaborative filtering recommendation algorithms by Sarwar
    http://portal.acm.org/citation.cfm?id=372071

    """

    def __init__(self, model, similarity="sim_pearson", neighborhood_size=None, model_size=None,
                 build_itemSim=True, with_preference=False):
        self.model = model
        self.itemPrefs = self._transformPrefs()
        self._similarity = Similarity(self.model, 'item', self.itemPrefs, similarity)
        self.neighborhood_size = neighborhood_size
        self.model_size = model_size
        self.with_preference = with_preference
        if build_itemSim:
            #TODO put a feedback telling to setup a itemSimilarity before call recommendations
            #TODO change the type of itemSimilarity to dict for speed purposes
            self.itemSimilarity = self.calculateSimilarItems(model_size)
        else:
            self.itemSimilarity = None

    def calculateSimilarItems(self, n=50):
        prefs = self.model
        # Create a dictionary of items showing which other items they
        # are most similar to.
        result = {}
        # Computes similarities for every item. Expensive computation

        # Invert the preference matrix to be item-centric
        itemPrefs = self.itemPrefs
        c = 0
        for item in itemPrefs:
            # Status updates for large datasets
            c += 1
            if c % 100 == 0: print "%d / %d" % (c, len(itemPrefs))
            # Find the most similar items to this one
            scores = self._topMatches(itemPrefs, item, n, self._similarity.similarity)
            result[item] = scores

        return result

    def _transformPrefs(self):
        prefs = self.model
        result = {}
        for person in prefs:
            for item in prefs[person]:
                result.setdefault(item, {})
                # Flip item and person
                result[item][person] = prefs[person][item]
        return result

    def getRecommendedItems(self, user, how_many=None):
        itemMatch = self.itemSimilarity
        if itemMatch is None:
            raise ValueError("Must build item similarity relation first.")
        prefs = self.model
        userRatings = prefs[user]
        scores = {}
        totalSim = {}
        
        # Loop over items rated by this user
        for (item, rating) in userRatings.items():

            # Loop over items similar to this one
            # all neighbors item-based
            neighborhood = self.most_similar_items(item, self.neighborhood_size)
            for (sim, neighbor) in neighborhood:

                # Ignore if this user has already rated this item
                if neighbor in userRatings:
                    continue
                # Weighted sum of rating times similarity
                scores.setdefault(neighbor, 0)
                scores[neighbor] += sim * rating
                # Sum of all the similarities
                totalSim.setdefault(neighbor, 0)
                totalSim[neighbor] += sim

        # Divide each total score by total weighting to get an average
        rankings = [(score / totalSim[item], item) for item, score in scores.items() if totalSim[item] != 0]

        # Return the rankings from highest to lowest
        rankings.sort()
        rankings.reverse()

        # Remove scores if with_preference is set to False
        if not self.with_preference:
            rankings = [item for score, item in rankings]

        if how_many is None:
            return rankings
        else:
            return rankings[0:how_many]

    def estimate_preference(self, user, target_item, **params):
        """
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        item_id:  int or string
            ID of item for which wants to find the estimated preference.

        Returns
        -------
        :rtype : float
        Return an estimated preference if the user has not expressed a
        preference for the item, or else the user's actual preference for the
        item. If a preference cannot be estimated, returns None.
        """

        prefs = self.model
        userRatings = prefs[user]
        scores = {}
        totalSim = {}
        scores.setdefault(target_item, 0.0)
        totalSim.setdefault(target_item, 0)

        #===== fixing estimate_preference

        # Loop over items rated by this user
        for (item, rating) in userRatings.items():

            # Loop over items similar to this one
            similar_items = self.most_similar_items(item, how_many=self.neighborhood_size)
            for (sim, item2) in similar_items:

                # Ignore if this user has already rated this item
                if item2 != target_item:
                    continue

                # Weighted sum of rating times similarity
                scores.setdefault(item2, 0)
                scores[item2] += sim * rating
                # Sum of all the similarities
                totalSim.setdefault(item2, 0)
                totalSim[item2] += sim

        result = scores[target_item] / float(totalSim[target_item]) if totalSim[target_item] != 0 else None

        return result

    def _estimate_my_items(self, user, **params):
        prefs = self.model
        itemMatch = self.itemSimilarity
        userRatings = prefs[user]
        scores = {}
        totalSim = {}
        # Loop over items rated by this user
        for (item, rating) in userRatings.items():
            # Loop over items similar to this one
            for (similarity, item2) in itemMatch[item]:
                # Pick only rated itens
                if item2 not in userRatings:
                    continue
                    # Weighted sum of rating times similarity
                scores.setdefault(item2, 0.0)
                scores[item2] += similarity * rating
                # Sum of all the similarities
                totalSim.setdefault(item2, 0)
                totalSim[item2] += similarity

        estimated_items = {}
        # Divide each total score by total weighting to get an average
        rankings = [(score // totalSim[item], item) for item, score in scores.items() if totalSim[item] != 0 ]
        for score, item in rankings:
            estimated_items[item] = score

        return estimated_items

    def all_other_items(self, user_id, **params):
        '''
        Parameters
        ----------
        user_id: int or string
                 User for which recommendations are to be computed.

        Returns
        ---------
        Return items in the `model` for which the user has not expressed
        the preference and could possibly be recommended to the user.

        '''
        raise NotImplementedError

    def _topMatches(self, prefs, person, n=5):
        """
        n determines the number of items to considere
        If n is none consider every item (item-item)
        """

        scores = [(self._similarity.calculate(person, other), other) for other in prefs if other != person]
        scores.sort()
        scores.reverse()
        return scores[0:n] if n is not None else scores

    def most_similar_items(self, item_id, how_many=None):
        """
        Return the most similar items to the given item, ordered
        from most similar to least.

        Parameters
        -----------
        item_id:  int or string
            ID of item for which to find most similar other items

        how_many: int
            Desired number of most similar items to find (default=None ALL)

        Return
        -----------
        :rtype : list
        scores: list
            List of tuples (s, i) where 's' is the similarity of the item 'i' with item 'item_id'
        """
#        scores = [(self.itemSimilarity[item_id][other], other) for other in self.itemSimilarity if other != item_id]
#        scores.sort()
#        scores.reverse()
#        return scores[0:how_many] if how_many is not None else scores

        return self.itemSimilarity[item_id][0:how_many] if how_many is not None else self.itemSimilarity[item_id]

    def recommended_because(self, user_id, item_id, how_many=None, **params):
        """
        Returns the items that were most influential in recommending a
        given item to a given user. In most implementations, this
        method will return items that the user prefers and that
        are similar to the given item.

        Parameters
        -----------
        user_id : int or string
            ID of the user who was recommended the item

        item_id: int or string
            ID of item that was recommended

        how_many: int
            Maximum number of items to return (default=None ALL)

        Returns
        ----------
        The list of items ordered from most influential in
        recommended the given item to least
        """
        raise NotImplementedError

    def calculate(self, item, other):

        sim = 0
        #TODO convert itemSimilarity to dict for better performance over list
        if self.itemSimilarity:
            for simi, neighbor in self.itemSimilarity[item]:
                if neighbor ==  other:
                    sim = simi
                    break
        else:
            sim = self._similarity.calculate(item, other)
        return sim


