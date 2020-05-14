# Discrete Input
# Chow Liu's Algorithm
# One parent
# Base Classifier: Naive Bayes
# Training Scheme: Real Value from dataset
# Select Root: Random
# One Chain


def create_classifiers(train, parents, topologicalSort):
    """
    This function creates classifiers for each class and link them according to
    dependency directed tree which is built from Chow & Liu's algorithm(MWST)
    :param train: training dataset
    :param parents: a dictionary that contains parent of each node
    :param topologicalSort: a topological sort of classifiers
    :return: a dictionary of trained classifiers
    """
    from sklearn.naive_bayes import BernoulliNB
    from read_inputs import create_training_set_for_bayesian_based_chain_classifier

    # A dictionary of classifiers for each class
    # {class0: classifier0, ...,classN: classifierN}
    classifiers = {classifier: None for classifier in topologicalSort}

    # for each class makes an gaussian naive bayes classifiers and train it
    for key in classifiers:
        classifiers[key] = BernoulliNB(alpha=1)

        # Create appropriate classifier
        trainingSet = create_training_set_for_bayesian_based_chain_classifier(train=train,
                                                         classOfClassifier=key,
                                                         parentClassOfClassifier=parents[key])

        # Train classifier
        classifiers[key].fit(trainingSet[0], trainingSet[1])

    return classifiers


def predict(classifiers, parents, topologicalSort, testSample):
    """
    Use classifiers according to their order in topological sort
    to predict labels of test sample.
    :param classifiers: a dictionary of trained classifiers
    :param parents: a dictionary that contains parent of each node
    :param topologicalSort: a topological sort of classifiers
    :param testSamples: A test sample which this function
            wants to predict its labels
    :return: list of labels
    """
    # a dictionary for holding prediction for classes
    # it supposed 0 and -1 [maybe probability * should be checked]
    predictions = {Class: None for Class in parents}

    for _class in topologicalSort:
        if parents[_class] != None:
            testSample.append(predictions[parents[_class]])

        label = classifiers[_class].predict([testSample])
        predictions[_class] = label[0]

        if parents[_class] != None:
            testSample.pop()

    return predictions


def tnbcc_DISCRETE(path):
    """
        This function uses other functions to read
        dataset, train classifiers and evaluate model
        with ten-fold cross validation.
        :param path: path of data-set
        :return: 10-fold accuracy of model
    """

    from mwst import chow_liu_tree
    from os.path import join
    from read_inputs import read_input_discrete
    from accuracy_of_model import ten_fold_cross_validation_without_scaling
    from dependencyTreeOfClasses import create_dataset_for_chow_liu_tree, \
        create_directed_tree_from_chow_liu_tree,\
        topological_sort_of_classifiers

    # read and scale train
    pathTrain = join(path, path+'.arff')
    train = read_input_discrete(pathTrain)

    print 'Train: ', len(train['train'])

    trainChowLiu = create_dataset_for_chow_liu_tree(dataset=train)
    print 'ChowLiu Training: ', trainChowLiu

    chowLiuTree = chow_liu_tree(dataset=trainChowLiu, n=len(trainChowLiu[0]))
    print 'ChowLiu tree edges: ', chowLiuTree.edges(data=True)

    root, directedDependency = create_directed_tree_from_chow_liu_tree(chowLiuTree=chowLiuTree)

    print 'Chow and Liu\'s directed tree nodes: ', directedDependency.nodes()
    print 'Chow and Liu\'s directed tree edges: ', directedDependency.edges()

    parents, topologicalSort = topological_sort_of_classifiers([root, directedDependency])

    print 'Parent list: ', parents
    print 'Topological Order: ', topologicalSort

    additionalArguments = {'parents': parents, 'topologicalSort': topologicalSort}
    TenFoldAcc = ten_fold_cross_validation_without_scaling(learningFunction=create_classifiers,
                                                           predictFuncion=predict, dataSet=train,
                                                           additionalArg=additionalArguments)

    return TenFoldAcc


if '__main__' == __name__:
    print tnbcc_DISCRETE('medical')