
from pymanopt.manifolds import Sphere
import pymanopt
import torch
import numpy as np 
import plotly.graph_objects as go 


sphere = Sphere(4)

random_points = []
random_normals = []





num_iters = 100
key = torch.Generator()
key.manual_seed(1234)
import scipy
for j in range(num_iters):
    random_point = sphere.random_point()
    random_points.append(random_point)
    while True:
        random_tangy_1 = sphere.random_tangent_vector(random_point).reshape(-1,1)
        random_tangy_2 = sphere.random_tangent_vector(random_point).reshape(-1,1)
        random_tangy_3 = sphere.random_tangent_vector(random_point).reshape(-1,1)
        

        
        Aug_matrix = np.vstack((random_tangy_1.T, random_tangy_2.T, random_tangy_3.T))
        
        if np.linalg.matrix_rank(Aug_matrix) == 3:
            orthog = scipy.linalg.null_space(Aug_matrix)
            if np.dot(orthog.flatten(), random_point.flatten()) > 0:
                random_normals.append(orthog)

        
        
            
outer_list = []
for i in range(num_iters):
    outer_list.append((random_normals[i]/np.linalg.norm(random_normals[i])).flatten()) # CHANGED TO AS TO BE USELESS TO PLOT
Normals = np.array(outer_list) # N x 4
Points = np.array(random_points) # N x 4




Fig = go.Figure()

trace_inner = go.Scatter3d(
    x = Points[: , 0],
    y = Points[: , 1],
    z = Points[: , 2],
    p = Points[:, 3],
    name = "inner",
    mode = "markers"
    # color = 'red'
)

trace_outer = go.Scatter3d(
    x = Normals[: , 0],
    y = Normals[: , 1],
    z = Normals[: , 2],
    p = Normals[:, 4]
    name = "outer",
    mode = "markers"
    # color = 'blue'
)

Fig.add_trace(trace_inner)
Fig.add_trace(trace_outer)

Fig.show()

nx = Normals[: , 0]
ny = Normals[: , 1]
nz = Normals[: , 2]
nw = Normals[:, 3]
x = Points[: , 0]
y = Points[: , 1]
z = Points[: , 2]
w = Points[:, 3]




#CREATING THE FILE

pre_boilerplate = [
    "ply",
    "format ascii 1.0",
    f"element vertex {num_iters}",
    "property float x",
    "property float y",
    "property float z",
    "property float nx",
    "property float ny",
    "property float nz",
    "end_header"
]

file_name = "SPHERE.ply"

with open(file_name, "w") as f:
    #f.write(line + '\n')
    for line in pre_boilerplate:
        f.write(line + '\n')
    for i in range(num_iters):
        f.write(f"{x[i]} {y[i]} {z[i]} {nx[i]} {ny[i]} {nz[i]}\n")


'''
need properties x,y,z nx,ny, nz so 6 in sum
'''
'''
ply format general 
format ascii 1.0           { ascii/binary, format version number }
comment made by Greg Turk  { comments keyword specified, like all lines }
comment this file is a cube
element vertex 8           { define "vertex" element, 8 of them in file }
property float x           { vertex contains float "x" coordinate }
property float y           { y coordinate is also a vertex property }
property float z           { z coordinate, too }
element face 6             { there are 6 "face" elements in the file }
property list uchar int vertex_index { "vertex_indices" is a list of ints }
end_header                 { delimits the end of the header }
0 0 0                      { start of vertex list }
0 0 1
0 1 1
0 1 0
1 0 0
1 0 1
1 1 1
1 1 0
4 0 1 2 3                  { start of face list }
4 7 6 5 4
4 0 4 5 1
4 1 5 6 2
4 2 6 7 3
4 3 7 4 0
'''