def mean_accuracy(predictions, dataset):
    """
    This functions Calculates mean accuracy of model
    which can be a binary relevance model or bayesian
    based chain.
    :param predictions for test samples:
    :param dataset: the dataset which its samples were used
    :return: accuracy
    """
    acc = {key: 0 for key in predictions[0]}

    for indexObs in range(0, len(dataset['train'])):
        for indexClass in range(0, dataset['numberOfLabels']):
            if int(predictions[indexObs][indexClass]) == int(dataset['train'][indexObs][dataset['classColumns'][indexClass]]):
                acc[indexClass] += 1

    sumOfDifferentLabel = 0.0
    for key in acc:
        acc[key] /= len(dataset['train'])+0.0
        sumOfDifferentLabel += acc[key]

        meanAccuracy = sumOfDifferentLabel / (len(acc)+0.0)

    print acc
    # Return Mean Accuracy
    return meanAccuracy


def ten_fold_cross_validation_without_scaling(learningFunction, predictFuncion, dataSet, additionalArg):
    """
    This function uses 10-fold-cv to evaluate learningFunction
    which can be SVM,NaiveBayes, and any other learning algorithm.
    :param learningFunction: Is a function that learns
            from data to predict label of new inputs.
            It can be any Machine Learning algorithms
            like TNBCC, TSVMCC and ...
    :param predictFuncion: Is a function that predicts labels of data
            according to the function which learned
            from data.
    :param argumentOFLearningFunction: is a list
        that contains necessary argument for
        learningFunction and predictFuncion.
    :param dataSet: training data
    :return: return average 10-fold cv error
    """
    from math import floor
    import copy as cp

    test = {'name': dataSet['name'],  # Name of Data set
               'attributes': cp.copy(dataSet['attributes']),  # Name of each columns (feature)
               'classColumns': cp.deepcopy(dataSet['classColumns']),  # Number of columns which are classes
               'numberOfLabels': dataSet['numberOfLabels'],  # Number of different labels
               'train': []  # Data, each row is an observation
               }
    train = {'name': dataSet['name'],  # Name of Data set
            'attributes': cp.copy(dataSet['attributes']),  # Name of each columns (feature)
            'classColumns': cp.deepcopy(dataSet['classColumns']),  # Number of columns which are classes
            'numberOfLabels': dataSet['numberOfLabels'],  # Number of different labels
            'train': []  # Data, each row is an observation
            }

    # average error on 10 folds
    averageAcc = 0

    # calculate size of each fold
    foldsize = int(floor(len(dataSet['train'])/10.0))

    # A list that contain 10 folds
    folds=[]

    # Divide dataSet to ten fold
    for fold in range(9):
         folds.append(dataSet['train'][fold*foldsize:(fold+1)*foldsize])
    folds.append(dataSet['train'][(10-1) * foldsize::])

    # Train and test learning function with 10 different forms
    for index1, i in enumerate(folds):
        # Test contains fold[i]
        test['train'] = cp.deepcopy(i)
        # Train contains all folds except fold[i]
        train['train'] = []
        for index2, j in enumerate(folds):
            if index2 != index1:
                train['train'] = train['train'] + cp.deepcopy(j)

        # Evaluate performance of learningFunction

        if additionalArg == None:
            # Create classifiers
            classifiers = learningFunction(train=train)
        else:
            # Create classifiers
            classifiers = learningFunction(train=train, **additionalArg)

        # for classifier in classifiers:
        # print classifiers[classifier].class_prior_

        predictions = []
        for index in range(0, len(test['train'])):
            if additionalArg == None:
                predictedLabels = predictFuncion(classifiers=classifiers,
                                      testSample=test['train'][index][0:test['classColumns'][0]])
            else:
                predictedLabels = predictFuncion(classifiers=classifiers,
                                                 testSample=test['train'][index][0:test['classColumns'][0]],
                                                 **additionalArg)

            predictions.append(cp.deepcopy(predictedLabels))

        mean_accuracy_of_chain = mean_accuracy(predictions=predictions, dataset=test)
        averageAcc += mean_accuracy_of_chain

    averageAcc /= 10.0

    return averageAcc



def ten_fold_cross_validation_with_scaling(learningFunction, predictFuncion, dataSet,additionalArg):
    """
    This function uses 10-fold-cv to evaluate learningFunction
    which can be SVM,NaiveBayes, and any other learning algorithm.
    :param learningFunction: Is a function that learns
            from data to predict label of new inputs.
            It can be any Machine Learning algorithms
            like TNBCC, TSVMCC, KNN, decisionTree, SVM and ...
    :param predictFuncion: Is a function that predict labels of data
            according to the function which learned
            from data.
    :param argumentOFLearningFunction: is a list
        that contains necessary argument of
        learningFunction.
    :param dataSet: training data
    :return: return average 10-fold cv accuracy
    """
    from math import floor
    import copy as cp
    from sklearn import preprocessing

    test = {'name': dataSet['name'],  # Name of Data set
               'attributes': cp.copy(dataSet['attributes']),  # Name of each columns (feature)
               'classColumns': cp.deepcopy(dataSet['classColumns']),  # Number of columns which are classes
               'numberOfLabels': dataSet['numberOfLabels'],  # Number of different labels
               'train': []  # Data, each row is an observation
               }
    train = {'name': dataSet['name'],  # Name of Data set
            'attributes': cp.copy(dataSet['attributes']),  # Name of each columns (feature)
            'classColumns': cp.deepcopy(dataSet['classColumns']),  # Number of columns which are classes
            'numberOfLabels': dataSet['numberOfLabels'],  # Number of different labels
            'train': []  # Data, each row is an observation
            }

    # average error on 10 folds
    averageAcc = 0

    # calculate size of each fold
    foldsize = int(floor(len(dataSet['train'])/10.0))

    # A list that contain 10 folds
    folds=[]

    # Divide dataSet to ten fold
    for fold in range(9):
         folds.append(dataSet['train'][fold*foldsize:(fold+1)*foldsize])
    folds.append(dataSet['train'][(10-1) * foldsize::])

    # Train and test learning function with 10 different forms
    for index1, i in enumerate(folds):
        # Test contains fold[i]
        test['train'] = cp.deepcopy(i)
        # Train contains all folds except fold[i]
        train['train'] = []
        for index2, j in enumerate(folds):
            if index2 != index1:
                train['train'] = train['train'] + cp.deepcopy(j)

        # scale train
        scaler = preprocessing.StandardScaler().fit(train['train'])
        trainCopy = cp.copy(train['train'])
        train['train'] = scaler.transform(train['train']).tolist()
        for indexObs, observation in enumerate(train['train']):
            for indexFeature in range(train['classColumns'][0],
                                          train['classColumns'][train['numberOfLabels'] - 1] + 1):
                train['train'][indexObs][indexFeature] = trainCopy[indexObs][indexFeature]

        # scale test
        testCopy = cp.copy(test['train'])
        test['train'] = scaler.transform(test['train']).tolist()
        for indexObs, observation in enumerate(test['train']):
            for indexFeature in range(test['classColumns'][0],
                                          test['classColumns'][test['numberOfLabels'] - 1] + 1):
                test['train'][indexObs][indexFeature] = testCopy[indexObs][indexFeature]

        # Evaluate performance of learningFunction

        if additionalArg == None:
            # Create classifiers
            classifiers = learningFunction(train=train)
        else:
            # Create classifiers
            classifiers = learningFunction(train=train, **additionalArg)

        # for classifier in classifiers:
        # print classifiers[classifier].class_prior_

        predictions = []
        for index in range(0, len(test['train'])):
            if additionalArg == None:
                predictedLabels = predictFuncion(classifiers=classifiers,
                                      testSample=test['train'][index][0:test['classColumns'][0]])
            else:
                predictedLabels = predictFuncion(classifiers=classifiers,
                                                 testSample=test['train'][index][0:test['classColumns'][0]],
                                                 **additionalArg)

            predictions.append(cp.deepcopy(predictedLabels))

        mean_accuracy_of_chain = mean_accuracy(predictions=predictions, dataset=test)
        averageAcc += mean_accuracy_of_chain

    averageAcc /= 10.0

    return averageAcc
