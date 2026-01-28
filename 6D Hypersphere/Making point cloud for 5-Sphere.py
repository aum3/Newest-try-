import numpy as np
import geometric_kernels
from geometric_kernels.spaces import Circle, ProductDiscreteSpectrumSpace, Hypersphere
from geometric_kernels.kernels import MaternGeometricKernel
import matplotlib as mpl
import matplotlib.pyplot as plt



#Defining the HYPERSPHERE
HYPERSPHERE = Hypersphere(dim = 5)


###Creating randm points
SEED = 333
np_random_key = np.random.RandomState(SEED)



num_samples = 150
_, random_points = HYPERSPHERE.random(np_random_key, num_samples) #n X 6  matrix

# Normalize points to ensure they lie exactly on the unit sphere
random_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)

print("Min norm:", np.min([np.linalg.norm(x) for x in random_points]))
print("Max norm:", np.max([np.linalg.norm(x) for x in random_points]))
print("Shape:", random_points.shape)





np.save("6D_HYPERSPHERE_POINT_CLOUD.npy", random_points)


