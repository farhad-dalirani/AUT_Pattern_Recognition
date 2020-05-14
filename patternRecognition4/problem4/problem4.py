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


# Part A
# Use range -5, 5 for drawing normal distribution N(0,1)
xRange = np.linspace(start=-5, stop=5, num=100)

# plot N(0,1)
plt.plot(xRange,
         [univariate_normal(mean=0, variance=1, x=x) for x in xRange],
          '--', label='N(0,1)')


# Part B,C
#draw 100 samples from N(0,1)
normalSamples = np.random.standard_normal(size=100)

# plot estimated distribution for different h
for h in [0.1, 0.3, 0.5, 0.75, 1, 5]:
    plt.plot(xRange,
             [estimate_px(dataset=normalSamples, x=x, h=h) for x in xRange],
             '-', label='h={}'.format(h))

plt.grid()
plt.legend()
plt.show()