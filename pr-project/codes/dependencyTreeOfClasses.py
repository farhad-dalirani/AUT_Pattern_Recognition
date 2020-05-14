"""
Funtions of this file build a directed tree according to Chow and Liu's tree.
Chow and Liu's tree implementation is in mwst.py
"""

def create_directed_tree_from_chow_liu_tree(chowLiuTree):
    """
    Create a directed tree from a Chow and Liu's tree(MWST) which is bi-directed.
    :param chowLiuTree: is an undirected graph. a networkx object.
    :return: return a pair (root, directedTree),
             root: is the root of directed tree
             directedTree: a directed tree which is built from a Chow and Liu's tree(MWST).
             the tree returned as a networkx.digraph object.
    """
    # A package for working with graphs
    import networkx as nx
    import random

    # Directed Tree which is built from Chow & Liu's
    # tree
    directedTree = nx.DiGraph()

    # Add nodes of Chow & Liu's to directed tree
    directedTree.add_nodes_from(chowLiuTree.nodes())

    # Get list of nodes of chowLiuTree
    nodesOfChowLiuTree = chowLiuTree.nodes()

    # A bool dictionary for preventing seeing a node more
    # than one time
    nodesFirstTime = {x:True for x in nodesOfChowLiuTree}

    # Select a root for tree;
    # *should add other manners for selecting root

    # Select root randomly
    root = random.choice(nodesOfChowLiuTree)

    # Select root, Node with most children
    #numberOfParent = [(chowLiuTree.neighbors(node), node) for node in nodesOfChowLiuTree]
    #sorted(numberOfParent)

    #candidateRoot = []
    #for index in range(0, len(numberOfParent)):
    #    if numberOfParent[index][0] == numberOfParent[len(numberOfParent)-1][0]:
    #        candidateRoot.append(numberOfParent[index][1])
    #root = random.choice(candidateRoot)

    # iterating Chow Liu by DFS
    stack = []
    stack.append(root)
    nodesFirstTime[root] = False

    while(len(stack) != 0):

        # pop element on the stack
        currentNode = stack.pop()

        # Add neighbors of node to stack and
        # add edge [currentNode, neighbor] for all neighbours
        # to directed tree
        for neighbor in chowLiuTree.neighbors(currentNode):
            # if node hasn't seen before
            if nodesFirstTime[neighbor] == True:
                directedTree.add_edge(currentNode, neighbor)
                stack.append(neighbor)
                nodesFirstTime[neighbor] = False

    # Return directed tree
    return (root, directedTree)


def topological_sort_of_classifiers(rootDirectedTree):
    """
    This function converts directed tree which is built from Chow & Liu's
    tree to a topological order of classes.
    :param rootDirectedTree:  a pair (root, directedTree),
             root: is the root of directed tree
             directedTree: a directed tree which is built from a Chow and Liu's tree(MWST).
             the tree returned as a networkx.digraph object.
    :return: a pair (parents, topologicalSort):
                parents: a dictionary that contains parent of each node,
                topologicalSort: a topological sort of classifiers
    """
    import networkx as nx

    # topology
    topologicalSort = nx.topological_sort(rootDirectedTree[1])

    parents = {rootDirectedTree[0]: None}
    for edge in rootDirectedTree[1].edges():
        parents[edge[1]] = edge[0]

    return (parents, topologicalSort)


def create_dataset_for_chow_liu_tree(dataset):
    """
    This function creates a dataset for chow and liu's MWST algorithm,
    it just contain classes of each observation in this form:

    [[class0 sample0, class1 sample0,...,classN sample0],
     [class0 sample1, class1 sample1,...,classN sample1],
     ...,
     [class0 sampleN, class1 sampleN,...,classN sampleN]]

    :param dataset:
    :return: appropriate dataset for creating chow & liu tree
    """
    import copy as cp
    newdataset = []
    for observation in dataset['train']:

        newObservation = []

        for index in range(dataset['classColumns'][0], len(observation)):
            newObservation.append(observation[index])

        newdataset.append(cp.copy(newObservation))

    # return proper dataset for finding chow and liu's tree
    return newdataset
