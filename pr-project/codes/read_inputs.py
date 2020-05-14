"""
This file contains two function for reading continuous and
discrete data sets.
"""

def read_input_discrete(filePath):
    """
    This function reads discrete datasets from file.
    :param filePath: path of data-set
    :return: a list in this form:
            [[x0 feature0, x0 feature1,..., x0 featureN, x0 class0, ..., x0 classM],
             [x1 feature1, x1 feature1,..., x1 featureN, x1 class0, ..., x1 classM],
             ...
             [xn feature0, xn feature1,..., xn featureN, xn class0, ..., xn classM]]
    """
    import copy as cp
    import numpy as np
    from sklearn import preprocessing

    # DataSet Description
    dataSet = {'name': '',            # Name of Data set
               'attributes': [],      # Name of each columns (feature)
               'classColumns': {},    # Number of columns which are classes
               'numberOfLabels': 0,   # Number of different labels
               'train': []            # Data, each row is an observation
               }

    # Read Train file
    # Read Train
    trainFile = open(filePath, 'r')
    train = []

    # For realizing we reach data part of dataset
    dataObserved = False

    # number of column
    numberOfColumn = 0
    numberOfClass = 0

    # read each line of dataset
    for line in trainFile:

        if  dataObserved == False:

            if '@relation' in line:
                nameOfDataset = line.split(' ')
                nameOfDataset = nameOfDataset[1]
                dataSet['name'] = filePath+'-'+nameOfDataset
                continue

            if '@attribute' in line:
                attributeName = line.split(' ')
                kind = attributeName[1]
                attributeName = attributeName[1]
                dataSet['attributes'].append(attributeName)

                if '.' in kind or 'Class' in kind or 'TAG_' in kind:
                    dataSet['classColumns'][numberOfClass]= numberOfColumn
                    numberOfClass += 1
                    dataSet['numberOfLabels'] = numberOfClass

                numberOfColumn += 1
                continue

            if '@data' in line:
                dataObserved = True
                continue

        else:
            # remove white spaces
            line = line.replace('\t', '')
            line = line.replace('\n', '')
            line = line.replace('{', '')
            line = line.replace('}', '')

            # split line by ',' for separating features
            features = line.split(',')

            # each line of dataset is an observation
            # [feature1,...,featureN, class i]
            observation = [0] * len(dataSet['attributes'])

            for feature in features:
                index_value = feature.split(' ')
                observation[int(float(index_value[0]))] = int(float(index_value[1]))

            # add observation to training set
            train.append(cp.copy(observation))


    # Shuffle Data *
    np.random.shuffle(train)

    dataSet['train'] = cp.copy(train)

    #for t in dataSet['train']:
    #    print t
    #    input()

    # Return dataSet
    return dataSet


def read_input_continues(filePath):
    """
        This function reads continues datasets from file.
        :param filePath: path of data-set
        :return: a list in this form:
                [[x0 feature0, x0 feature1,..., x0 featureN, x0 class0, ..., x0 classM],
                 [x1 feature1, x1 feature1,..., x1 featureN, x1 class0, ..., x1 classM],
                 ...
                 [xn feature0, xn feature1,..., xn featureN, xn class0, ..., xn classM]]
        """

    import copy as cp
    import numpy as np
    from sklearn import preprocessing

    # DataSet Description
    dataSet = {'name': '',            # Name of Data set
               'attributes': [],      # Name of each columns (feature)
               'classColumns': {},    # Number of columns which are classes
               'numberOfLabels': 0,   # Number of different labels
               'train': []            # Data, each row is an observation
               }

    # Read Train file
    # Read Train
    trainFile = open(filePath, 'r')
    train = []

    # For realizing we reach data part of dataset
    dataObserved = False

    # number of column
    numberOfColumn = 0
    numberOfClass = 0

    # read each line of dataset
    for line in trainFile:

        if  dataObserved == False:

            if '@relation' in line:
                nameOfDataset = line.split(' ')
                nameOfDataset = nameOfDataset[1]
                dataSet['name'] = filePath+'-'+nameOfDataset
                continue

            if '@attribute' in line:
                attributeName = line.split(' ')
                kind = attributeName[2]
                attributeName = attributeName[1]
                dataSet['attributes'].append(attributeName)

                if '{' in kind:
                    dataSet['classColumns'][numberOfClass]= numberOfColumn
                    numberOfClass += 1
                    dataSet['numberOfLabels'] = numberOfClass

                numberOfColumn += 1
                continue

            if '@data' in line:
                dataObserved = True
                continue

        else:
            # remove white spaces
            line = line.replace(' ', '')
            line = line.replace('\t', '')
            line = line.replace('\n', '')
            # split line by ',' for seperating features
            features = line.split(',')

            # each line of dataset is an observation
            # [feature1,...,featureN, class i]
            observation = []
            for index, feature in enumerate(features):
                if index < dataSet['classColumns'][0]:
                    # for features
                    observation.append(float(feature))
                else:
                    # for classes
                    observation.append(float(feature))

            # add observation to training set
            train.append(cp.copy(observation))


    # Shuffle Data *
    np.random.shuffle(train)

    dataSet['train'] = cp.copy(train)

    #for t in dataSet['train']:
    #    print t
    #    input()

    # Return dataSet
    return dataSet


def create_training_set_for_binary_relevance_classifier(train, classOfClassifier):
    """
    This function gets a multi-label dataset and prepares a new train dataset
    for each classifier.
    Training set for each classifiers in binary relevance just
    consists of features and the class of classifier.

    :param train: training data
    :param classOfClassifier: determines a class that classifiers wanted to
                                predict data belongs to it or not
    :return:return a pair (new training set, actual class of training set)
            new training set: a new training set for the classifier in this form
            [[feature0,..., featureN, class parent],
              ...,
             [feature0,..., featureN, class parent]]

             actual class of training set: which is a list in this form:
            [class of observation 0 {0,1}, ..., class of observation N {0,1}]
    """
    import copy as cp

    # [[feature0,..., featureN, class parent],
    #   ...,
    # [feature0,..., featureN, class parent]]
    newTrain = []

    # [class of new observation0,...,class of new observationN]
    yTrain = []

    # from each observation of training set make a new observation
    # that contains features and parent class as feature
    for observation in train['train']:
        newobservation = []
        for index, feature in enumerate(observation):

            if index < train['classColumns'][0]:
                # add features to new training set
                newobservation.append(feature)
            else:
                break

        # Corresponding class of new observation, for training
        yTrain.append(int(observation[train['classColumns'][classOfClassifier]]))
        # New Observation
        newTrain.append(cp.copy(newobservation))

    #print '>>>>>', newTrain
    #print '/////', yTrain
    return [newTrain, yTrain]


def create_training_set_for_bayesian_based_chain_classifier(train, classOfClassifier, parentClassOfClassifier):
    """
    This function gets a mulilabel dataset and prepares a new train dataset
    for each classifier.
    Training set for each classifiers in a bayesian based chain manner
    consists of features, output of parent classifier as feature and
    the class of classifier.

    :param train: training data
    :param classOfClassifier: determines a class that classifiers wanted to
                                predict data belongs to it or not
    :param parentClassOfClassifier: parent of classifier
    :return:return a pair (new training set, actual class of training set)
            new training set: a new training set for the classifier in this form
            [[feature0,..., featureN, class parent],
              ...,
             [feature0,..., featureN, class parent]]

             actual class of training set: which is a list in this form:
            [class of observation 0 {0,1}, ..., class of observation N {0,1}]
    """
    import copy as cp

    # [[feature0,..., featureN, class parent],
    #   ...,
    # [feature0,..., featureN, class parent]]
    newTrain = []

    # [class of new observation0,...,class of new observationN]
    yTrain = []

    # from each observation of training set make a new observation
    # that contains features and parent class as feature
    for observation in train['train']:
        newobservation = []
        for index, feature in enumerate(observation):

            if index < train['classColumns'][0]:
                # add features to new training set
                newobservation.append(feature)
            elif parentClassOfClassifier != None:
                # omit classes and and parent class as a feature
                newobservation.append(observation[train['classColumns'][parentClassOfClassifier]])
                break

        # Corresponding class of new observation, for training
        yTrain.append(int(observation[train['classColumns'][classOfClassifier]]))
        # New Observation
        newTrain.append(cp.copy(newobservation))

    return [newTrain, yTrain]

