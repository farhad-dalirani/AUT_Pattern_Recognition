import matplotlib.pylab as plt
import numpy as np

x1 = np.matrix([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
x1 = np.transpose(x1)
x2 = np.matrix([[9,  10], [6, 8], [9, 5], [8, 7], [10, 8]])
x2 = np.transpose(x2)

########################
# A
########################
plt.figure()
plt.plot(x1[0].tolist()[0], x1[1].tolist()[0], 'b*', label='Class X1')
plt.plot(x2[0].tolist()[0], x2[1].tolist()[0], 'r*', label='Class X2')
plt.gca().set_aspect('equal', adjustable='box')

########################
# B
########################
# Mean of x1
mean1 = np.sum(x1, axis=1) / np.shape(x1)[1]
x1MinesMean1 = x1 - mean1
print('Mean Class 1:\n {} \n'.format(mean1))

# Mean of x2
mean2 = np.sum(x2, axis=1) / np.shape(x2)[1]
x2MinesMean2 = x2 - mean2
print('Mean Class 2:\n {} \n'.format(mean2))

# mean of all elements
meanTotal = (np.sum(x1, axis=1)+np.sum(x2, axis=1))/(np.shape(x1)[1]+np.shape(x2)[1])
print('Mean of all elements:\n {} \n'.format(mean2))

# Calculate scatter 1: s1 = cov1 * (n1 - 1)
s1 = np.cov(x1MinesMean1) * (np.shape(x1MinesMean1)[1]-1)
print('Scatter Class 1:\n {} \n'.format(s1))

# Calculate scatter 1: s1 = cov1 * (n1 - 1)
s2 = np.cov(x2MinesMean2) * (np.shape(x2MinesMean2)[1]-1)
print('Scatter Class 2:\n {} \n'.format(s2))

# Calculate S within
Sw = s1 + s2
print('Scatter within(Sw):\n {} \n'.format(Sw))

# Calculate W = inverse sw * (mean1 - mean2)
w = np.linalg.inv(Sw) * (mean1 - mean2)
print('W:\n {} \n'.format(w))

# Normalize w
w = w / np.sqrt(w.item((0, 0))**2 + w.item((1, 0))**2)


# print and draw w
print('Normalized W:\n {} \n'.format(w))
plt.arrow(meanTotal.item((0, 0)),
          meanTotal.item((1, 0)),
          w.item((0,0)) * 2, w.item((1,0)) * 2,
          0.5, linewidth=1, head_width=0.5, color='green', alpha=0.4)


plt.title('X1 AND X2, w= [{}, {}]'.format(
    round(w.item(0), 4), round(w.item(1), 4)))


plt.xlim(0, 15)
plt.ylim(-3, 12)
plt.grid()

########################
# C
########################
# Project data to w : y = transpose(w) * x
y1 = np.transpose(w) * x1
y2 = np.transpose(w) * x2

print('Projection of Class1:\n {} \n'.format(y1))
print('Projection of Class2:\n {} \n'.format(y2))

plt.figure()
plt.plot(y1[0].tolist()[0], [0]*len(y1[0].tolist()[0]),
         'b*', label='Projection of class X1')
plt.plot(y2[0].tolist()[0], [0]*len(y2[0].tolist()[0]),
         'r*', label='Projection of class X2')
plt.ylim(-5, 5)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Projection of Samples on W: [{},{}]'.format(
    round(w.item(0), 4), round(w.item(1), 4)))
plt.legend()
plt.show()
