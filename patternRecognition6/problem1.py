import numpy as np
import matplotlib.pylab as plt
import random

aTranspose = np.array([0.2, 1, -1])

w1 = np.array([[0.1, 0.7],
                [0.3, 0.55],
                [0.45, 0.15],
                [0.6, 0.3]])

w2 = np.array([[0.15, 1],
                [0.35, 0.95],
                [0.7, 0.65],
                [0.9, 0.45]])

# Augmented
w1Aug = np.array([np.append(np.array([1]), point, axis=0) for point in w1])
w2Aug = np.array([np.append(np.array([1]), point, axis=0) for point in w2])

# Negative w2
w2Aug = -1 * w2Aug

# learning rate
mu = 0.49

#########################################
# A
#########################################
plt.plot([x0[0] for x0 in w1], [x1[1] for x1 in w1], 'b*', label='Class 1')
plt.plot([x0[0] for x0 in w2], [x1[1] for x1 in w2], 'r^', label='Class -1')

# Draw Decision Boundary
x0Range = np.linspace(start=0, stop=1, num=100)
plt.plot(x0Range, [x0 + aTranspose[0] for x0 in x0Range], label='Initial')
# Draw W
plt.arrow(x0Range[50], x0Range[50] + aTranspose[0], aTranspose[1]*0.1, aTranspose[2]*0.1,
          0.5, linewidth=1, head_width=0.01, color='green', alpha=0.4)


#########################################
# B
#########################################
c = 0
# update an misclassified point
for point in w1Aug:
    # Calculate value of g(y) = transpose(a) . y
    gPoint = aTranspose * point
    if np.sum(gPoint) > 0:
        continue
    else:
        c = c + 1
        print('Misclassified Point: {}'.format(point))
        print('Weight before update: {}'.format(aTranspose))
        aTranspose = aTranspose + mu * point
        print('Weight after update: {}\n'.format(aTranspose))
        break

# Draw Decision Boundary
x0Range = np.linspace(start=0, stop=1, num=100)
plt.plot(x0Range,
         [x0*(-1*aTranspose[1]/aTranspose[2]) + (-1*aTranspose[0]/aTranspose[2]) for x0 in x0Range],
                     label=str(c))# Draw W
plt.arrow(x0Range[50],
          x0Range[50] * (-1 * aTranspose[1] / aTranspose[2]) + (-1 * aTranspose[0] / aTranspose[2]),
          aTranspose[1]*0.1, aTranspose[2]*0.1,
          0.5, linewidth=1, head_width=0.01, color='green', alpha=0.4)

#########################################
# C
#########################################
# update weight for misclassified points for 4 times
b = 0
while c < 5 and b < 10000:
    b = b + 1

    classes = random.choice([1, -1])
    index = random.choice([0, 1, 2, 3])

    if index == 1:
        point = w1Aug[index]
    else:
        point = w2Aug[index]

    # Calculate value of g(y) = transpose(a) . y
    gPoint = aTranspose * point
    if np.sum(gPoint) < 0:
        # If it classified correctly, go to next point
        c = c + 1
        print('Misclassified Point: {}'.format(point))
        print('Weight before update: {}'.format(aTranspose))
        aTranspose = aTranspose + mu * point
        print('Weight after update: {}\n'.format(aTranspose))

        # Draw Decision Boundary
        x0Range = np.linspace(start=0, stop=1, num=100)
        plt.plot(x0Range, [x0*(-1*aTranspose[1]/aTranspose[2]) + (-1*aTranspose[0]/aTranspose[2]) for x0 in x0Range],
            label=str(c))
        # Draw W
        plt.arrow(x0Range[50],
        x0Range[50] * (-1 * aTranspose[1] / aTranspose[2]) + (
            -1 * aTranspose[0] / aTranspose[2]), aTranspose[1]*0.1, aTranspose[2]*0.1,
                0.5, linewidth=1, head_width=0.01, color='green', alpha=0.4)


plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


