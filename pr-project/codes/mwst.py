"""
Approximating Discrete Probability Distributions with Dependence Trees
By C.K. CHOW AND C.N.LIU
IEEE TRANSACTIONS ON INFORMATION THEORY,VOL.IT-14,NO.3,MAY 1968
"""

import networkx as nx
import numpy as np
from collections import defaultdict


def marginal_probabilities(dataset, u):
    """
    :param data-set: Data-set
    :param u: Index of u'th features
    :return: The marginal probabilities for the u'th features of the data-set.
            return is a default dictionary object.
    """
    # For each value of feature u'th calculate its probability
    # in entire data-set
    values = defaultdict(float)
    prob = 1. / len(dataset)
    for x in dataset:
        values[x[u]] += prob

    # Return probabilities of different value of u'th features as
    # a default dictionary
    return values


def marginal_pair_probabilities(dataset, u, v):
    """
    :param X: Data-set
    :param u: Index u'th features
    :param v: Index v'th features
    :return: The marginal probabilities for the u'th and v'th features of the data-set.
            return is a default dictionary object.
    """
    if u > v:
        u, v = v, u

    # For each pair of values of features u and v calculate its probability
    # in entire data-set
    values = defaultdict(float)
    prob = 1. / len(dataset)
    for x in dataset:
        values[(x[u], x[v])] += prob
    return values


def mutual_information(dataset, u, v):
    """
    :param dataset: Data-set
    :param u: Index of u'th features
    :param v: Index of v'th features
    :return: return mutual information of u'th and v'th features
    """
    if u > v:
        u, v = v, u
    # Calculate probabilities of different values of
    # u'th feature in data-set X
    marginal_u = marginal_probabilities(dataset, u)

    # Calculate probabilities of different values of
    # v'th feature in data-set X
    marginal_v = marginal_probabilities(dataset, v)

    # Calculate probabilities of different values of
    # pair u'th feature and v'th feature in data-set X
    marginal_uv = marginal_pair_probabilities(dataset, u, v)

    # Calculate mutual information according to marginal
    # distributions
    mutualInformation = 0.0
    for x_u, p_x_u in marginal_u.iteritems():
        for x_v, p_x_v in marginal_v.iteritems():
            if (x_u, x_v) in marginal_uv:
                p_x_uv = marginal_uv[(x_u, x_v)]
                mutualInformation += p_x_uv * (np.log(p_x_uv) - np.log(p_x_u) - np.log(p_x_v))

    # return mutual information
    return mutualInformation


def chow_liu_tree(dataset, n):
    """
    This function makes a chow liu's tree. First it calculates
    mutual information among all features, then it multiply all
    mutual information by -1 and at last it builds chow & liu's
    tree by constructing minimum spanning tree of a complete graph
    which its nodes are features and its edges are mutual information
    which were multiplied by -1.

    :param dataset: Data-set
    :param n: Number of features
    :return: An undirected tree which is a networkx object.
    """
    # Construct complete graph of nodes(features)
    # and edges (-1 * mutual information of each two features)
    completeGraph = nx.Graph()
    for v in xrange(n):
        completeGraph.add_node(v)
        for u in xrange(v):
            completeGraph.add_edge(u, v, weight=-mutual_information(dataset, u, v))

    # Construct chow & liu's tree by applying minimum spanning tree
    # on the complete graph
    chowLiuUndirectedTree = nx.minimum_spanning_tree(completeGraph)

    # Return chow & liu's tree
    return chowLiuUndirectedTree

