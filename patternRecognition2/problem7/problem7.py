########################################################################
# Problem 7 - a, b
########################################################################


def c1(x):
    """
    Determine value of x in uniform distribution (2, 4)
    :param x: x is a given point
    :return: value of point x in U(2, 4)
    """
    if(x >= 2 and x <= 4):
        return 1/2
    else:
        return 0


def c2(x):
    """
    Determine value of x in exponential distribution (lambda = 1)
    :param x: x is a given point
    :return: value of point x in exponential(lambda = 1)
    """
    import numpy as np
    return np.exp(-1 * x)


import numpy as np
import matplotlib.pyplot as plt

priorC1 = priorC2 = 0.5

# Generate 200 point for drawing curves between 0 & 10
x = np.linspace(0, 10, 200)

# Generate density of class1
density1 = [c1(point) * priorC1 for point in x]

# Generate density of class2
density2 = [c2(point) * priorC2 for point in x]

# Plot densities
plt.plot(x, density1, 'r-', label='p(x|c1)*P(c1)')
plt.plot(x, density2, 'b-', label='p(x|c2)*P(c2)')

# Plot decision boundaries
y = np.linspace(-0.05, 0.55, 100)
plt.plot([2]*100, y, 'g--', label='Decision boundary x=2')
plt.plot([4]*100, y, 'y--', label='Decision boundary x=4')

#plt.grid()
plt.legend()
plt.xlim(0, 10)

########################################################################
# Problem 7 - c
########################################################################
def approximate_error(a, b, priorOfClassC2):
    """
    This function approximates Integral e^(-x)
    according to below relations
    taylor e^x: e^x = 1 + x + (x^2)/2! + (x^3)/3! + ...
    taylor e^(-x): e^(-x) = 1 - x + (x^2)/2! - (x^3)/3! + ...

    Integral e^(-x) = Integral 1 - x + (x^2)/2! - (x^3)/3! + ...
    which is equal to
    Integral e^(-x) = [x - (x^2)/2! + (x^3)/3! - (x^4)/4! + ...] on interval[a,b]

    :param a: start of integral interval
    :param b: end of integral interval
    :param priorOfClassC2: prior of class c2 for multipling it to area under
            e^(-x)
    :return: approximate integral
    """
    # Calculate Integral according to above formula
    factoral = 1
    valueOfIntegral = 0
    sign = 1
    for i in range(1, 21):
        valueOfIntegral += sign * (b**i - a**i) / factoral
        sign *= -1
        factoral *= i+1

    return valueOfIntegral * priorOfClassC2


plt.title('Bayes Error is: {}'.format(approximate_error(a=2, b=4, priorOfClassC2=0.5)) )
plt.show()


print("> Approximate Error is: ",
        approximate_error(a=2, b=4, priorOfClassC2=0.5))
