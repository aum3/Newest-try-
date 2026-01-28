import numpy as np
import geometric_kernels
from geometric_kernels.spaces import Circle, ProductDiscreteSpectrumSpace
from geometric_kernels.kernels import MaternGeometricKernel
import matplotlib as mpl
import matplotlib.pyplot as plt



#Defining the torus
torus = ProductDiscreteSpectrumSpace(Circle(), Circle(), Circle())


###Creating randm points
SEED = 333
np_random_key = np.random.RandomState(SEED)


num_samples = 150
_, random_points = torus.random(np_random_key, num_samples) #n X 2  matrix


x1 = np.cos(random_points[:, 0:1])
y1 = np.sin(random_points[:, 0:1])
x2 = np.cos(random_points[:, 1:2])
y2 = np.sin(random_points[:, 1:2])
x3 = np.cos(random_points[:, 2:3])
y3 = np.sin(random_points[:, 2:3])

random_points = np.hstack((x1, y1, x2, y2, x3, y3))  # n x 6 matrix




np.save("6D_TORUS_POINT_CLOUD.npy", random_points)


