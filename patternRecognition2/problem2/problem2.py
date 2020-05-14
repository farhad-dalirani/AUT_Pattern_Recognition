########################################################################
# Problem 2 - Needed Function
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
# Problem 2 - a
########################################################################
def g1(sampleVector):
    """
    This function get a vector of sample and uses
    'discriminant_function' function to calculates
    value of discriminant function for p(x|w1) when
    P(w1) = 0.6, mean = (4, 16) and Sigma = 4*I
    :param sampleVector: a vector of sample
    :return: a vector that contains corresponding discriminant
            value for each input sample.
    """
    import numpy as np

    means = [4, 16]
    pw1 = 0.6
    sigma = np.identity(2) * 4

    return discriminant_function(mu=means, covariance=sigma,
                                 priorProbability=pw1, sampleVector=sampleVector)


def g2(sampleVector):
    """
    This function get a vector of sample and uses
    'discriminant_function' function to calculates
    value of discriminant function for p(x|w2) when
    P(w1) = 0.4, mean = (16, 4) and Sigma = 4*I
    :param sampleVector: a vector of sample
    :return: a vector that contains corresponding discriminant
            value for each input sample.
    """
    import numpy as np

    means = [16, 4]
    pw2 = 0.4
    sigma = np.identity(2) * 4

    return discriminant_function(mu=means, covariance=sigma,
                                 priorProbability=pw2, sampleVector=sampleVector)


########################################################################
# Problem 2 - Needed Function
########################################################################
import numpy as np
import matplotlib.pyplot as plt

# First manner: Finding equation of x according g1(x) = g2(x)
# which is equal to "x = [ x1,  x1-(1/3)*(Ln(3/2))]"
# feature 1 and 2 points on decision boundary
x1 = np.linspace(-50, 50, 200)
x2 = [x-(1/3)*(np.log(3/2)) for x in x1]

# Plot curve
plt.figure()
plt.plot(x1,x2, label="Decision Boundary")
plt.grid()
plt.legend()
plt.title("Decision Boundary")
plt.xlim(-50, 50)

# Second manner: Shows areas with g1(x) > g2(x) with
# different color that areas which g1(x) <= g2(x)
def draw_decision_boundary(plot_range):
    """
    This function draws discriminant function of to different
    normal distribution which are given in problem 2 and also,
    It draws decision boundary
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

    # Discriminant function for normal distribution 1
    g1Value = g1(points)
    g1Value.shape = plot_resolution, plot_resolution

    # Discriminant function for normal distribution 2
    g2Value = g2(points)
    g2Value.shape = plot_resolution, plot_resolution

    # Creating a figure and three equal subplot in it
    fig, axes = pl.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes.ravel()
    for ax in axes.ravel():
        ax.set_aspect("equal")

    ax1.pcolormesh(X, Y, g1Value)
    ax2.pcolormesh(X, Y, g2Value)

    # Determining decision boundary
    ax3.pcolormesh(X, Y, g1Value > g2Value)


draw_decision_boundary((-50, 50))
plt.show()
