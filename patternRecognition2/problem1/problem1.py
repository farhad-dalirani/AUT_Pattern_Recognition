########################################################################
# Problem 1
########################################################################

import matplotlib.pyplot as plt
import numpy as np


# p(x|w1) and p(x|w2):
# densities
def xw1(x):
    if x >=0 and x <= 1:
        return 2*x
    else:
        return 0


def xw2(x):
    if x >=0 and x <= 1:
        return 2-2*x
    else:
        return 0


densities = [xw1, xw2]

########################################################################
# Problem 1- a
########################################################################
# Generate 200 point for drawing curves between (-1,2)
x = np.linspace(-1,2,num=200)

# Plot both densities
plt.plot(x, list(map(densities[0], x)), label='P(x|w1)', linestyle='-')
plt.plot(x, list(map(densities[1], x)), label='P(x|w2)', linestyle='-')


########################################################################
# Problem 1- b
########################################################################
# By finding x according to P(w1) and P(w2) from below equation:
# p(x|w1)*P(w1) - p(x|w2)*P(w2) = 0
# we can easily find decision boundary.
# when p(x|w1) is 2x in (0,1) and
# p(x|w2) is 2-2x in (0,1) then
# decision boundary is x = P(w2)/( P(w1)+P(w2))
def calculate_decision_boundary(pw1, pw2):
    """
    This function finds decision boundary of p(x|w1) = 2x in (0,1) and
    p(x|w2) = 2-2x in (0,1)
    :param pw1: prior  P(w1)
    :param pw2: prior  P(w2)
    :return: return a number which indicate
            decision boundary(x0 = P(w2)/( P(w1)+P(w2))
    """
    return pw2/(pw1 + pw2)


# Generate 200 point for drawing decision boundary (-0.5,2.5)
y = np.linspace(-0.5, 2.5, num=200)

# Plot decision boundary when P(w1) = P(w2) = 0.5
plt.plot([calculate_decision_boundary(pw1=.5, pw2=0.5)]*200, y,
         label='Decision Boundary when P(w1)={},P(w2)={}'.format(0.5, 0.5)
         , linestyle='--')


########################################################################
# Problem 1- c
########################################################################
def error_of_classifier(pw1, pw2, x0):
    """
        This function finds errors of classifier when
        p(x|w1) = 2x in (0,1) and
        p(x|w2) = 2-2x in (0,1)
        :param pw1: prior  P(w1)
        :param pw2: prior  P(w2)
        :param x0: x0 is decision boundary
        :return: return a number which indicate
                error of classifier
    """
    return (pw1 * (x0 * (2*x0)) / 2) + (pw2 * ((1-x0) * (2-2*x0)) / 2)


# Calculate Error of classifier when P(w1) = P(w2) = 0.5
err = error_of_classifier(pw1=0.5, pw2=0.5, x0=calculate_decision_boundary(pw1=0.5, pw2=0.5))
print("Error of classifier when P(w1)={}, P(w2)={} is {}".format(
    0.5, 0.5, err))

########################################################################
# Problem 1- d
########################################################################
# Plot decision boundary when P(w1) = 0.3, P(w2) = 0.7
plt.plot([calculate_decision_boundary(pw1=.3, pw2=0.7)]*200, y,
         label='Decision Boundary when P(w1)={},P(w2)={}'.format(0.3, 0.7)
         , linestyle='--')


plt.title("Densities And Decision Boundaries, Error when P(w1)=P(w2)=0.5 is: {}".format(err))
plt.xlabel('X')
plt.ylabel('Value Of Densities')
plt.ylim((-0.75,4))
plt.legend()
plt.show()
