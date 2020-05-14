from computerExercise1 import discriminant_function
##############################################
# Train Data
##############################################
w1 = [
    [-5.01, -8.12, -3.68],
    [-5.43, -3.48, -3.54],
    [1.08, -5.52, 1.66],
    [0.86, -3.78, -4.11],
    [-2.67, 0.63, 7.39],
    [4.94, 3.29, 2.08],
    [-2.51, 2.09, -2.59],
    [-2.25, -2.13, -6.94],
    [5.56, 2.86, -2.26],
    [1.03, -3.33, 4.33]]

w2 = [
    [-0.91, -0.18, -0.05],
    [1.30, -2.06, -3.53],
    [-7.75, -4.54, -0.95],
    [-5.47, 0.50, 3.92],
    [6.14, 5.72, -4.85],
    [3.60, 1.26, 4.36],
    [5.37, -4.63, -3.65],
    [7.18, 1.46, -6.66],
    [-7.39, 1.17, 6.30],
    [-7.50, -6.32, -0.31]]

w3 = [
    [5.35, 2.26, 8.13],
    [5.12, 3.22, -2.66],
    [-1.34, -5.31, -9.87],
    [4.48, 3.42, 5.19],
    [7.11, 2.39, 9.21],
    [7.17, 4.33, -0.98],
    [5.75, 3.97, 6.65],
    [0.77, 0.27, 2.41],
    [0.90, -0.43, -8.71],
    [3.52, -0.36, 6.43]]

# A dictionary of different classes and labels
classes = {'w1': w1, 'w2': w2, 'w3': w3}

# All samples without label
data = w1 + w2 + w3


##############################################
# Computer exercise 3- needed functions from computer exercise 2
##############################################
def covariance_matrix(samples):
    """
    This function gets some samples and return its covariance
    :param samples: each col represents a variable,
     with observations in the row
    :return: Corresponding covariance matrix
    """
    import numpy as np
    samplesArray = np.asarray(samples)
    samplesArray = np.transpose(samplesArray)

    # Cov function calculates covariance it this way:
    # X_new(k) = X(k) - mu
    # B = [X_new(1),X_new(2),...,X_new(n)]
    # Cov = (1/(n-1))*B*transpose(B)
    return np.cov(samplesArray).tolist()


def mean_vector(X):
    """
    :param X: each col represents a variable,
     with observations in the row
    :return: return a list that contains mean of each col
    """
    import numpy as np
    means = np.mean(X, axis=0)
    return means.tolist()


def mean_and_covariance(samples):
    """
    This function gets some samples and then it finds
    corresponding Covariance and mean of their normal distribution.
    :param samples: samples.each col represents a variable,
     with observations in the row
    :return: a dictionary with keys: {'means': , 'covariance'}
    """
    dic = {}
    dic['means'] = mean_vector(samples)
    dic['covriance'] = covariance_matrix(samples)
    return dic


def dichotomizer( listOfnormalDis, x):
    """
    Labeling sample x by a label which has maximum g
    :param listOfnormalDis: is list that contains dictionaries,
        each dictionary contains these keys: {'means','covariance', 'name','prior'} that are
        properties of a normal distribution
    :param x: an input which this function determine it belongs to normal distribution
        one or two
    :return: If x belongs to normal distribution y output is
    like: (x, y['name'])
    """

    # Calculate value of discriminant function for sample x and each normal distribution
    g = []
    for normalDist_i in listOfnormalDis:
        if normalDist_i['prior'] != 0:
            g.append(discriminant_function(mu=normalDist_i['means'], covariance=normalDist_i['covriance'],
                              priorProbability=normalDist_i['prior'], sampleVector=x))
        else:
            g.append(float('-inf'))

    # Determine sample x belongs to which class
    return (x, listOfnormalDis[g.index(max(g))]['name'])


def empirical_training_error(labeledSamples, pridictedSampled):
    """
    This function counts number of misclassified samples
    :param labeledSamples: is a dictionary that contains n keys w1, w2, ..., wn
        each wi is assigned to a list of samples
    :param pridictedSampled: corresponding prediction for each labeled sample,
        It's a list likes this: [ [sample1, predicted class], ...,[sample2, predicted class]
    :return: percentage of misclassified samples
    """
    # check predicted label to sample i is True or not
    misclassified = 0
    for i in pridictedSampled:
        # Check samples exist where prediction says or not
        if i[0] not in labeledSamples[i[1]]:
            misclassified = misclassified + 1

    return misclassified / len(predictByclassifier) * 100


def bhattacharyya_bound(normalDist1, normalDist2):
    """
    This measure shows similarities between two
    distribution.
    :param normalDist1: is dictionary that contains
            these keys {'means','covariance', 'name','prior'}
    :param normalDist2: is dictionary that contains
            these keys {'means','covariance', 'name','prior'}
    :return: return a scalar
    """
    import numpy as np

    # Convert data to proper form
    mean1 = np.asmatrix(normalDist1['means'])
    mean2 = np.asmatrix(normalDist2['means'])
    cov1 = np.asmatrix(normalDist1['covriance'])
    cov2 = np.asmatrix(normalDist2['covriance'])

    # Average of two covariances
    covAve = (cov1 + cov2) / 2

    # Subtract of means
    meanSub = mean2 - mean1

    # Calculate the bound
    bound = (np.sqrt(normalDist1['prior'] * normalDist2['prior'])) * (
            np.exp(-1 * ((1/8) * (meanSub * np.linalg.inv(covAve) * np.transpose(meanSub)) +
                         (1/2) * np.log(
                             np.linalg.det(covAve)/np.sqrt(np.linalg.det(cov1) *
                                                           np.linalg.det(cov2))
                                         ))))
    #return bound
    return bound.tolist()[0][0]


##############################################
# Computer exercise 3
##############################################
from copy import deepcopy

printFeatures = ['X1','X1, X2','X1, X2, X3']

# Use X1 feature then X1,X2 then X1,X2,X3
for useFirstNfeature in range(1,4):
    # Prepare data
    # Make a copy of classes and just keep wanted features
    # of samples.
    classesFx1 = deepcopy(classes)
    for key in classesFx1:
        # Iterate all samples of class key
        for index, element in enumerate(classesFx1[key]):
            # just keep first useFirstNfeature
            classesFx1[key][index] = element[0:useFirstNfeature]

    # Make a copy of data and just keep wanted features
    dataFx1 = deepcopy(data)
    for index, x in enumerate(dataFx1):
        dataFx1[index] = x[0:useFirstNfeature]

    # Because of classifying  w1 and w3, just use w1 and w3 samples
    dataFx1 = dataFx1[0:10] + dataFx1[20:30]

    # Priories of classes
    priories = {'w1': 0.5, 'w2': 0, 'w3': 0.5}

    # Find normal distribution of each class
    classOrder = {}
    normalDists = []
    for key in classesFx1:
        # Find a normal distribution for class key
        normalDists.append( mean_and_covariance(classesFx1[key]))
        # Set class name of normal distribution
        normalDists[len(normalDists)-1]['name'] = key
        # Set class prior of normal distribution
        normalDists[len(normalDists)-1]['prior'] = priories[key]
        classOrder[key] = len(normalDists)-1

    # Find label of each sample
    predictByclassifier = []
    for x in dataFx1:
        # Determine class of unlabeled data x
        # and add the result of classifier to a list
        predictByclassifier.append( dichotomizer(normalDists, x))

    # Find empirical training error
    print("Computer exercises 3 (Features {}):".format(printFeatures[useFirstNfeature-1]))
    print("Error: ",
          empirical_training_error(classesFx1,predictByclassifier))

    print('Bhattacharyya bound for w1 & w3:{:f}\n'.format(
          bhattacharyya_bound(normalDists[classOrder['w1']], normalDists[classOrder['w3']])))
