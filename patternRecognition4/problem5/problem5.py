import numpy as np
import matplotlib.pylab as plt

def univariate_normal(mean, variance, x):
    """
    value of a point in univariate normal distribution
    :param mean:
    :param variance:
    :param x:
    :return:
    """
    return (1 / (np.sqrt(2 * np.pi * variance)) *
            np.exp((-1 / (2 * variance)) * ((x - mean) ** 2)))


def phi(x, xi, h):
    """
    Calculate window function at point xi
    which its center is x
    :param x: center of parzen window
    :param xi: given point
    :param h:
    :return: value of parzen window
    """
    import numpy as np
    return (1/np.sqrt(2*np.pi))*np.exp(-((x-xi)/h)*((x-xi)/h)/(2))


def estimate_px(dataset, x, h):
    """
    This function use parzen window estimator to
    estimate probability at point x
    :param dataset: list of point
    :param x: input x
    :param h:
    :return: return probability
    """
    allParzenWindow = 0
    for sample in dataset:
        allParzenWindow += phi(x=sample, xi=x, h=h)

    #calculate and return parzen estimate
    return allParzenWindow / (len(dataset)*h)


print('It takes several seconds, ...')
# Use range 0, 55 for drawing normal distribution N(20,5) ,N(35,5)
# (for drawing-not calculating estimated probability)
xRange = np.linspace(start=0, stop=55, num=300)


# for different number of samples estimate distribution
for sampleSize in [100, 1000]:

    # plot N(25,5)
    plt.plot(xRange,
             [univariate_normal(mean=25, variance=5, x=x) for x in xRange],
             'b--', label='N(0,1)')

    # plot N(35,5)
    plt.plot(xRange,
             [univariate_normal(mean=35, variance=5, x=x) for x in xRange],
             'b--', label='N(0,1)')

    #draw samples from N(25,5)
    normalSamples1 = np.random.normal(loc=25, scale=np.sqrt(5), size=sampleSize)

    #draw samples from N(35,5)
    normalSamples2 = np.random.normal(loc=35, scale=np.sqrt(5), size=sampleSize)

    # concatinate sample list 1 and 2
    normalSamples = normalSamples1.tolist() + normalSamples2.tolist()

    # plot estimated distribution for different h
    for h in [0.01, 0.1, 1, 10]:
        plt.plot(xRange,
                 [estimate_px(dataset=normalSamples, x=x, h=h) for x in xRange],
                 '-', label='h={}'.format(h))

    plt.grid()
    plt.legend()
    plt.show()