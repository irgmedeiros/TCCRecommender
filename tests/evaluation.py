from metrics.classes import userCfEvaluator, itemCfEvaluator
from recommenders.classes import ItemBasedRecommender, UserBasedRecommender
from recommenders import recommendations
import cPickle as pickle
import numpy as np
import scipy.spatial.distance as ssd
from similarities.classes import Similarity
#from scikits.crab.models.classes import MatrixPreferenceDataModel
#from scikits.crab.recommenders.knn.classes import ItemBasedRecommender
#from scikits.crab.similarities.basic_similarities import ItemSimilarity
#from scikits.crab.recommenders.knn.item_strategies import ItemsNeighborhoodStrategy
#from scikits.crab.metrics.pairwise import euclidean_distances
from recommenders.classes import UserBasedRecommender

def pearson_correlation(X, Y):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.

    This correlation implementation is equivalent to the cosine similarity
    since the data it receives is assumed to be centered -- mean is 0. The
    correlation may be interpreted as the cosine of the angle between the two
    vectors defined by the users' preference values.

    Parameters
    ----------
    X: array of shape (n_samples_1, n_features)

    Y: array of shape (n_samples_2, n_features)

    Returns
    -------
    distances: array of shape (n_samples_1, n_samples_2)

    Examples
    --------
    >>> from scikits.crab.metrics.pairwise import pearson_correlation
    >>> X = [[2.5, 3.5, 3.0, 3.5, 2.5, 3.0],[2.5, 3.5, 3.0, 3.5, 2.5, 3.0]]
    >>> # distance between rows of X
    >>> pearson_correlation(X, X)
    array([[ 1.,  1.],
           [ 1.,  1.]])
    >>> pearson_correlation(X, [[3.0, 3.5, 1.5, 5.0, 3.5,3.0]])
    array([[ 0.39605902],
           [ 0.39605902]])
    """
    # should not need X_norm_squared because if you could precompute that as
    # well as Y, then you should just pre-compute the output and not even
    # call this function.
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XY = ssd.cdist(X, Y, 'correlation', 2)

    return 1 - XY


def generateItemModel(item_itemSim, n=None):
    itemModel = item_itemSim.copy()
    for item in itemModel:
        itemModel[item] = itemModel[item][0:n] if n is not None else itemModel[item]

    return itemModel


def save_item_item_matrix(prefs, similarity="adjusted_cosine"):
    print "Pre calculating item-tem matrix with "+similarity+". It May take a while. Please Wait."
    fsim = file('item_item_Sim('+similarity+').pkl', 'wb')
    recsys = ItemBasedRecommender(model=prefs,
                                similarity=similarity,
                                neighborhood_size=None,
                                model_size=None,
                                build_itemSim=True,
                                with_preference=False)
    itemSim = recsys.itemSimilarity
    print "Done."

    pickle.dump(itemSim, fsim, True)
    fsim.close()


def save_user_user_matrix(prefs, similarity="adjusted_cosine"):
    print "Pre calculating user-user matrix with "+similarity+". It May take a while. Please Wait."
    fsim = file('user_user_Sim('+similarity+').pkl', 'wb')
    recsys = UserBasedRecommender(prefs, similarity,None,True,False)
    userSim = recsys.userSimilarity
    print "Done."
    pickle.dump(userSim, fsim, True)
    fsim.close()


def save_adj_cosine_matrix(prefs):
    def topMatches(prefs, person, n, similarity):
        scores = [(similarity.calculate(person, other), other)
                  for other in prefs if other != person]
        scores.sort()
        scores.reverse()
        return scores[0:n]

    # Create a dictionary of items showing which other items they
    # are most similar to.
    result = {}

    print "Pre calculating item-tem matrix. It May take a while. Please Wait."
    itemPrefs = recommendations.transformPrefs(prefs)
    sim = Similarity(prefs, 'item', itemPrefs, similarity='adjusted_cosine')
    fsim = file('item_item_Sim(adj_consine).pkl', 'wb')

    c = 0
    for item in itemPrefs:
        # Status updates for large datasets
        c += 1
        if c % 50 == 0: print "%d / %d" % (c, len(itemPrefs))
        # Find the most similar items to this one
        scores = topMatches(itemPrefs, item, len(itemPrefs), sim)
        result[item] = scores

    print "Done."
    pickle.dump(result, fsim, True)
    fsim.close()


def save_report(report):
    with open("report.txt", "a") as myfile:
        for element in report:
            myfile.write(element +": " + str(report[element]['mae']) + " 0," + str(report[element]['mae'])[2:] +"\n")
        myfile.write("\n")


def main():
    prefs = recommendations.loadMovieLens()

    mode = "user"
    sim = "adjusted_cosine"
    save = True

    if mode == "user":
        if save:
            save_user_user_matrix(prefs, sim)
        pickleFile = open('user_user_Sim('+sim+').pkl', 'rb')
        userUser = pickle.load(pickleFile)
        pickleFile.close()
#
        neighbor_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200]
        evaluations = {}

        for idx, size in enumerate(neighbor_sizes):
            recsys = UserBasedRecommender(model=prefs,similarity=sim,neighborhood_size=30, build_userSim=False, with_preference=True)
#            userSim = generateItemModel(userUser, size)
#            recsys.userSimilarity = userUser

            evaluator = userCfEvaluator()
#            print "Evaluating the UserBased Recommender with neighborhood size of " + str(size) + " using " + sim
            report = evaluator.evaluate(recsys, neighbor_sizes=neighbor_sizes, metric='mae', sampling_users=0.8)
            print "done"
            save_report(report)
            print report
#            evaluations['recsys'+str(idx)] = metric
#            print "UserBased Report:"
#            print "MAE:", metric["mae"], "size", size, "similarity", sim

    if mode == "item":
        if save:
            save_item_item_matrix(prefs, sim)
        pickleFile = open('item_item_Sim('+sim+').pkl', 'rb')
        itemItem = pickle.load(pickleFile)
        pickleFile.close()

        model_sizes = [25, 50, 75, 100, 125, 150, 175, 200, None]
        evaluations = {}

        for idx, size in enumerate(model_sizes):
            neighborhood = None

            recsys = ItemBasedRecommender(model=prefs,
                                similarity=sim,
                                neighborhood_size=neighborhood,
                                model_size=size,
                                build_itemSim=False,
                                with_preference=True)
            itemSim = generateItemModel(itemItem, size)
            recsys.itemSimilarity = itemSim

#            mostsim = recsys.most_similar_items("Adventures of Pinocchio, The (1996)")
            estim = recsys.estimate_preference('1', '12 Angry Men (1957)')


            evaluator = itemCfEvaluator()
            print "Evaluating the ItemBased Recommender with" + " neighborhood size of " + \
                  (str(neighborhood) if neighborhood else "All") + " and with model size of " + \
                  (str(size) if size else "All") + " using " + sim + ""

            metric = evaluator.evaluate(recsys, metric='mae', sampling_users=None, sampling_ratings=0.80, permutation=True)
            print "done"
            evaluations['recsys'+str(idx)] = metric
            print "ItemBased Report:"
            print "MAE:", metric["mae"], "size", size, "similarity", sim

    print evaluations


if __name__ == '__main__':
    main()