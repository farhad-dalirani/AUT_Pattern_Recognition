import numpy as np
import matplotlib.pylab as plt


############################################################
# 3-q5-c
############################################################

def normalDist(mu, var, x):
    """
    Calculate value of normal distribution at x
    :param mu: mean of normal distribution
    :param var: variance of normal distribution
    :param x: a plist of points
    :return: value of points of x in normal distribution
    """
    res = []
    for point in x:
        res.append((1/np.sqrt(2*np.pi*var)) * np.exp(((point-mu)**2)/(-2*var)))
    return res

# As I've shown in part b of the problem, variance T is
# equal p(1-p)/d
# probability of xi|w1 to be 1
p = 0.6

for d in [11, 111]:
    variance = p * (1-p)/d

    # Generate 1000 point between -1 and 2
    t = np.linspace(start=-1, stop=2, num=1000)

    # p(t|w1)
    pw1 = normalDist(mu=p, var=variance, x=t)

    # p(t|w2)
    pw2 = normalDist(mu=1-p, var=variance, x=t)

    # plot p(t|w1), p(t|w2)
    plt.figure()
    plt.plot(t, pw1, 'r', label="p(T|w1)")
    plt.plot(t, pw2, 'b', label="p(T|w2)")
    # plot T* = 0.5
    plt.plot([0.5]*100,
             np.linspace(start=-1, stop=3, num=100),
             'g--', label="T=0.5")

    plt.title('d = {}'.format(d))
    plt.xlabel('T')
    plt.ylabel('P(T|Wi)')
    plt.legend()
    plt.grid()
    plt.show()