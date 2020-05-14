########################################################################
# Problem 3- needed functions and given data set
########################################################################


def covariance_matrix(samples):
    """
    This function gets some samples and return its covariance matrix
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
    dic['covariance'] = covariance_matrix(samples)
    return dic


def discriminant_function(mu, covariance, priorProbability, sampleVector):
    """
    This function gets an vector of samples and calculates
    value of discriminant function for each.

    :param mu: a vector of means (list, couple, ...)
    :param covariance: covariance matrix (list, couple, ...)
    :param sampleVector: vectors of samples (list, couple, ...)
    :param priorProbability: prior probability (list, couple, ...)
    :return: A vector of discriminant of each sample in the sample vector (np.array)
    """
    import numpy as np

    # Convert inputs to numpy.matrix()
    Mu = np.matrix(mu)
    Cov = np.matrix(covariance)
    inputVector = np.matrix(sampleVector)

    # Dimension of given normal distribution
    d = Cov.shape[0]

    # Inverse of covariance matrix
    invCov = np.linalg.inv(Cov)

    # The part of discriminant function which is equal for different input samples
    discriminant_value_part2 = -(d/2)*np.log(2*np.pi)-\
                           (1/2)*np.log(np.linalg.det(Cov))+np.log(priorProbability)

    # Output
    outputVector = []

    # Calculating value of discriminant function for all input samples
    for i in range(inputVector.shape[0]):
        # Calculating value of discriminant function for input sample i
        sampleOutput = (-(1/2) * (inputVector[i] - Mu) * invCov * (inputVector[i] - Mu).transpose())+\
                       discriminant_value_part2
        outputVector.append(sampleOutput)

    #return an array of discriminant function
    return np.array(outputVector)


def draw_decision_boundary_2d_normal(mu1, cov1, prior1, mu2, cov2, prior2, plot_range):
    """
    This function draws discriminant function of to different
    normal distribution and also, It draws decision boundary
    :param mu1: means vector of distribution 1
    :param cov1: covariance matrix of distribution 1
    :param prior1: prior probability of distribution 1
    :param mu2: means vector of distribution 2
    :param cov2: covariance matrix of distribution 2
    :param prior2: prior probability of distribution 2
    :param plot_range: range is a tuple (like: (-2,2)) that determines
            range of plot which decision boundary will be drawn on it
    :return: none
    """
    import pylab as pl
    import numpy as np

    # This determines how many points make the plot.
    # Number of points on plot: plot_resolution * plot_resolution
    plot_resolution = 300

    # Make points of plot
    X, Y = np.mgrid[plot_range[0]:plot_range[1]:plot_resolution*1j,
           plot_range[0]:plot_range[1]:plot_resolution*1j]

    # Concatenate X,Y
    points = np.c_[X.ravel(), Y.ravel()]

    # Inverse of matrix cov1, for preventing reparation of computation
    invC = np.linalg.inv(cov1)

    # Discriminant function for normal distribution 1
    g1 = discriminant_function(mu1, cov1, prior1, points)
    g1.shape = plot_resolution, plot_resolution

    # Inverse of matrix cov1, for preventing repartition of computation
    invC = np.linalg.inv(cov2)

    # Discriminant function for normal distribution 2
    g2 = discriminant_function(mu2, cov2, prior2, points)
    g2.shape = plot_resolution, plot_resolution

    # Creating a figure and three equal subplot in it
    fig, axes = pl.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes.ravel()
    for ax in axes.ravel():
        ax.set_aspect("equal")

    ax1.pcolormesh(X, Y, g1)
    ax2.pcolormesh(X, Y, g2)

    # Determining decision boundary
    ax3.pcolormesh(X, Y, g1 > g2)

    pl.show()


# Given Data set
w1 = [(0, 0), (0, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
w2 = [(6, 9), (8, 9), (9, 8), (9, 9), (9, 10), (8, 11)]

classes = {1: w1, 2: w2}

########################################################################
# Problem 3- a
########################################################################
classesPrior = {}
numberOfAllSamples = 0
# Count number of all samples in all classes
for key in classes:
    numberOfAllSamples += len(classes[key])

# calculate and print prior probability of each class
for key in classes:
    classesPrior[key] = len(classes[key])/numberOfAllSamples
    print('Prior of w{} is :{}'.format(key, classesPrior[key]))


########################################################################
# Problem 3- b
########################################################################
# Find mean and covariance of each class
classesMeanAndCov = {}
for key in classes:
    # This function return mean and cov matrix in a dictionary
    # data structure, {'means': , 'covariance': }
    classesMeanAndCov[key] = mean_and_covariance(classes[key])

# print mean an covariance for each class
for key in classesMeanAndCov:
    print('W{}:\nmeans is: {},\ncovariance matrix is: {}'.format(
        key, classesMeanAndCov[key]['means'], classesMeanAndCov[key]['covariance']))


########################################################################
# Problem 3- C
########################################################################
# Draw decision boundary which separate class1 and class2
print("Drawing decision boundary takes several second, ...")
draw_decision_boundary_2d_normal(classesMeanAndCov[1]['means'],
                                 classesMeanAndCov[1]['covariance'],
                                 classesPrior[1],
                                 classesMeanAndCov[2]['means'],
                                 classesMeanAndCov[2]['covariance'],
                                 classesPrior[2],
                                 (-100, 100)
                                 )

