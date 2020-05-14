w1 = [[0.42, -0.087, 0.58],
[-0.2, -3.3, -3.4],
[1.3, -0.32, 1.7],
[0.39, 0.71, 0.23],
[-1.6, -5.3, -0.15],
[-0.029, 0.89, -4.7],
[-0.23, 1.9, 2.2],
[0.27, -0.3, -0.87],
[-1.9, 0.76, -2.1],
[0.87, -1.0, -2.6]]

w2 = [[-0.4, 0.58, 0.089],
[-0.31, 0.27, -0.04],
[0.38, 0.055, -0.035],
[-0.15, 0.53, 0.011],
[-0.35, 0.47, 0.034],
[0.17, 0.69, 0.1],
[-0.011, 0.55, -0.18],
[-0.27, 0.61, 0.12],
[-0.065, 0.49, 0.0012],
[-0.12, 0.054, -0.063]]

w3 = [[0.83, 1.6, -0.014],
[1.1, 1.6, 0.48],
[-0.44, -0.41, 0.32],
[0.047, -0.45, 1.4],
[0.28, 0.35, 3.1],
[-0.39, -0.48, 0.11],
[0.34, -0.079, 0.14],
[-0.3, -0.22, 2.2],
[1.1, 1.2, -0.46],
[0.18, -0.11, -0.49]]
###############################################################
# Computer Exercise 1
###############################################################
def MLE_mu(observationMatrix):
    """
    Calculate mean of input observation matrix by MLE
    MLE mu = 1/n * (X1+X2+...+Xn) , Xi is an observation
    in dataset
    :param observationMatrix:
    :return: estimated mu
    """
    import numpy as np

    # Convert to numpy array for matrix operation
    observationMatrix_np= np.array(observationMatrix)

    # inital sum of observations
    sumOfObservation = np.array([0]*len(observationMatrix_np[0]))

    # Calculate sum of observations
    for observaton in observationMatrix_np:
        sumOfObservation = sumOfObservation + observaton

    # divide by number of observation
    sumOfObservation /= len(observationMatrix)

    # return mean
    return sumOfObservation.tolist()


def MLE_var(observationMatrix, mu):
    """
    Calculate variance of input observation matrix by MLE
    MLE covariance = 1/n * ((X1-mu)(X1-mu)T+(X2-mu)(X2-mu)T+...+(Xn-mu)(Xn-mu)T)
            , Xi is an observation
    in dataset
    :param observationMatrix:
    :param mu: mean of observations which is obtained by MLE_mu
    :return: estimated variance
    """
    import numpy as np

    # Convert to numpy matrix for matrix operation
    observationMatrix_np= np.matrix(observationMatrix)

    # Convert to numpy matrix
    mu_np = np.matrix(mu)

    # inital sum of (Xi-mu)(Xi-mu)T
    sumOfdiff = np.array([0]*len(observationMatrix_np[0]))

    # Calculate sum of (Xi-mu)(Xi-mu)T
    for observaton in observationMatrix_np:
        sumOfdiff = sumOfdiff + \
                    (np.transpose(observaton-mu_np)*(observaton-mu_np))

    # divide by number of observation
    sumOfdiff /= len(observationMatrix)

    # return mean
    return sumOfdiff


###############################################################
# Computer Exercise 1 - a
###############################################################
# Calculate mean and variance for each feature of w1
w1_f0 = []
w1_f1 = []
w1_f2 = []
for observation in w1:
    w1_f0.append(observation[0:1])
    w1_f1.append(observation[1:2])
    w1_f2.append(observation[2:3])

meanf0 = MLE_mu(w1_f0)
meanf1 = MLE_mu(w1_f1)
meanf2 = MLE_mu(w1_f2)

varf0 = MLE_var(w1_f0, meanf0)
varf1 = MLE_var(w1_f1, meanf1)
varf2 = MLE_var(w1_f2, meanf2)

print("Part A:")
print("Mean feature0 ={}\nMean feature1 ={}\nMean feature2 ={}\n".format(
    meanf0, meanf1, meanf2))

print("Var feature0 ={}\nVar feature1 ={}\nVar feature2 ={}\n".format(
    varf0, varf1, varf2))

###############################################################
# Computer Exercise 1 - b
###############################################################
# Calculate mean and variance for each two features of w1
w1_f01 = []
w1_f02 = []
w1_f12 = []
for observation in w1:
    w1_f01.append(observation[0:1]+observation[1:2])
    w1_f02.append(observation[0:1]+observation[2:3])
    w1_f12.append(observation[1:2]+observation[2:3])

meanf01 = MLE_mu(w1_f01)
meanf02 = MLE_mu(w1_f02)
meanf12 = MLE_mu(w1_f12)

varf01 = MLE_var(w1_f01, meanf01)
varf02 = MLE_var(w1_f02, meanf02)
varf12 = MLE_var(w1_f12, meanf12)

print("Part B:")
print("Mean feature01 ={}\nMean feature02 ={}\nMean feature12 ={}\n".format(
    meanf01, meanf02, meanf12))

print("Covariance Matrix feature01 =\n{}\nCovariance Matrix feature02 =\n{}\nCovariance Matrix feature12 =\n{}\n".format(
    varf01, varf02, varf12))


###############################################################
# Computer Exercise 1 - C
###############################################################
# Calculate mean and variance for each feature of w1
w1_f012 = []
for observation in w1:
    w1_f012.append(observation)

# Calculate mu
meanf012 = MLE_mu(w1_f012)

# Calculate variance
varf012 = MLE_var(w1_f012, meanf012)

print("Part C:")
print("Mean feature012 ={}\n".format(
    meanf012))

print("Covariance Matrix feature012 =\n{}\n".format(
    varf012))

###############################################################
# Computer Exercise 1 - d
###############################################################
# Calculate mean and variance for each and all feature of w2
import numpy as np
w2_f0 = []
w2_f1 = []
w2_f2 = []
for observation in w2:
    w2_f0.append(observation[0:1])
    w2_f1.append(observation[1:2])
    w2_f2.append(observation[2:3])

meanf0 = MLE_mu(w2_f0)
meanf1 = MLE_mu(w2_f1)
meanf2 = MLE_mu(w2_f2)

varf0 = MLE_var(w2_f0, meanf0)
varf1 = MLE_var(w2_f1, meanf1)
varf2 = MLE_var(w2_f2, meanf2)

print("Part D:")
print("Mean feature0 ={}\nMean feature1 ={}\nMean feature2 ={}\n".format(
    meanf0, meanf1, meanf2))

print("Variance Matrix feature0 ={}\nCovariance Matrix feature1 ={}\nCovariance Matrix feature2 ={}\n".format(
    varf0, varf1, varf2))

mean012 = MLE_mu(w2)
covariance = MLE_var(w2, mean012)

covariance = covariance.tolist()
for i, obj1 in enumerate(covariance):
    for j, obj2 in enumerate(obj1):
        if i != j:
            covariance[i][j] = 0

print("Mean feature012 ={}\n".format(
    mean012))
print("Covariance feature012(assumed diagonal) =\n{}\n".format(
    np.matrix(covariance)))
