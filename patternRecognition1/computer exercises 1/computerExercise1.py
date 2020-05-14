########################################################################
# Computer exercise 1-a
########################################################################
def d_normal_distribution(mu, covMat, n):
    """
    D-dimensional normal distribution function,
       it draws samples from a D-dimensional normal distribution.
     :param
       (mu)mean vector,
      (covMat)covariance matrix,
      (n)number of samples that should be drawn.
     :return Output:
       each entry out[i,j,...,:] is an D-dimensional value drawn from normal distribution.
    An example of this function:
        d_normal_distribution(mu=[0,0],covMat=[[1, 0],[0, 100]], n=10)
    """
    import numpy as np
    # Use multivariate normal distribution of numpy package
    return np.random.multivariate_normal(mean=mu, cov=covMat, size=n, check_valid='ignore')


########################################################################
# Computer exercise 1-b
########################################################################
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


########################################################################
# Computer exercise 1-c
########################################################################
def euclidean_distance(x, y):
    """
    This function calculates euclidean distance between to
    arbitrary point.
    :param x: a point that is a list or couple like [x1, x2, ...,xn]
    :param y: a point that is a list or couple like [y1, y2, ...,yn]
    :return:
    """
    import numpy as np
    from math import sqrt

    # Convert point to numpy array
    x_array = np.array(x)
    y_array = np.array(y)

    return sqrt(np.dot(x_array-y_array, x_array-y_array))


########################################################################
# Computer exercise 1-d
########################################################################
def mahalanobis_distance(mu, covariance, x):
    """
    This function calculates Mahalanobis distance between
    a point and mean of a normal distribution
    :param mu: mean of the normal distribution
    :param covariance: covariance of the normal distribution
    :param x: input point that we want to find the distance of it
    :return: a scalar value that indicate Mahalanobis distance
    """
    import numpy as np
    from math import sqrt

    # Convert inputs to numpy.matrix()
    Mu = np.matrix(mu)
    Cov = np.matrix(covariance)
    point = np.matrix(x)

    # Calculate and return Mahalanobis distance
    return sqrt((point - Mu) * np.linalg.inv(Cov) * (point - Mu).transpose())

########################################################################
# Computer exercise 1- Extra function
########################################################################
def draw_decision_boundary_2d_normal( mu1, cov1, priori1, mu2, cov2, priori2, plot_range):
    """
    This function draws discriminant function of to different
    normal distribution and also, it draws decision boundary
    :param mu1: vector of means of distribution 1
    :param cov1: covariance matrix of distribution 1
    :param priori1: priori probability of distribution 1
    :param mu2: vector of means of distribution 2
    :param cov2: covariance matrix of distribution 2
    :param priori2: priori probability of distribution 2
    :param range: range is a tuple (like: (-2,2)) that determines
            range of plot which decision boundary will be drawn on it
    :return: none
    """
    import pylab as pl
    import numpy as np

    # This determines how many points make the plot.
    # Number of points on plot: plot_resolution * plot_resolution
    plot_resolution = 150

    # Make points of plot
    X, Y = np.mgrid[plot_range[0]:plot_range[1]:plot_resolution*1j, plot_range[0]:plot_range[1]:plot_resolution*1j]
    # Concat X,Y
    points = np.c_[X.ravel(), Y.ravel()]

    # Inverse of matrix cov1, for preventing repeating computation
    invC = np.linalg.inv(cov1)
    # Discriminant function for normal distribution 1
    g1 = discriminant_function(mu1, cov1, priori1, points)
    g1.shape = plot_resolution, plot_resolution

    # Inverse of matrix cov1, for preventing repeating computation
    invC = np.linalg.inv(cov2)
    # Discriminant function for normal distribution 2
    g2 = discriminant_function(mu2, cov2, priori2, points)
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


########################################################################
# Computer exercise 1- call functions for testing
########################################################################

# Test Computer exercise 1-a
#print(d_normal_distribution(mu=[0,2,0],covMat=[[1, 0, 2],[6,0, 100],[6,50, 100]], n=10))

# Test Computer exercise 1-b
#mu_1 = [0.1, 0.1]
#cov_1 = [[1.1, 0.3], [0.2, 0.8]]
#priori_1 = 0.5
#print(discriminant_function(mu_1, cov_1, priori_1, [[0.1,0.1], [5,5]]))

# Test Computer exercise 1-c
#print(euclidean_distance([1, 2, 3, 4],[1, 2, 4, 3]))

# Test Computer exercise 1-d
#mu0 = [3, 3]
#cov0 = [[1.1, 0.3],[0.3, 1.9]]
#x = [1.0, 2.2]
# Result should be: 1.9162463307205573
#print( mahalanobis_distance(mu0, cov0, x))

# Test Computer exercise 1- extra function: draw_decision_boundary_2d_normal()
#mu1 = [0.1, 0.1]
#mu2 = [1.1, 0.8]
#cov1 = [[1.1, 0.3], [0.2, 0.8]]
#cov2 = [[0.8, 0.5], [0.8, 2.1]]
#priori1 = 0.5
#priori2 = 0.5
#plot_range = (-10,10)
#draw_decision_boundary_2d_normal(mu1,cov1,priori1,mu2,cov2,priori2, plot_range)


