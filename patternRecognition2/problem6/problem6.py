########################################################################
# Problem 6- data
########################################################################

w1 = [
        [1.5, 0], [1, 1], [2, 1], [0.5, 2], [1.5, 2], [2.5, 2], [2, 3]
     ]
w2 = [
        [-2.5, -1], [-1, -1], [1, -1], [0.5, -0.5], [1.5, -0.5],
        [-1.5, 0], [-0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [-2, 1]
     ]


########################################################################
# Problem 6- needed functions
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


def draw_decision_boundary_2d_normal(mu1, cov1, prior1, mu2, cov2, prior2, plot_range, samplesw1, samplesw2):
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
    plot_resolution = 200

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

    # Draw samples on plot
    # Prepare samples of w1 and w2 for drawing
    w1 = [[],[]]
    for element in samplesw1:
        w1[0].append(element[0])
        w1[1].append(element[1])
    w2 = [[], []]
    for element in samplesw2:
        w2[0].append(element[0])
        w2[1].append(element[1])

    # Plot samples of w1
    ax3.plot(w1[0], w1[1], 'r*')

    # Plot samples of w2
    ax3.plot(w2[0], w2[1], 'bo')

    pl.grid()
    pl.show()


########################################################################
# Problem 6-
########################################################################

# mean and covariance matrix w1:
normal_dist_w1 = mean_and_covariance(w1)

# mean and covariance matrix w2:
normal_dist_w2 = mean_and_covariance(w2)

# Find Prior probabilities of w1 and w2
prior1 = len(w1) / (len(w1)+len(w2))
prior2 = len(w2) / (len(w1)+len(w2))

print('Drawing takes several seconds, Wait ...')
# Draw decision boundary and samples
draw_decision_boundary_2d_normal(mu1=normal_dist_w1['means'], cov1=normal_dist_w1['covariance'], prior1= prior1,
                                 mu2=normal_dist_w2['means'], cov2=normal_dist_w2['covariance'], prior2= prior2,
                                 plot_range=(-7.5, 5),
                                 samplesw1= w1,
                                 samplesw2= w2
                                 )
