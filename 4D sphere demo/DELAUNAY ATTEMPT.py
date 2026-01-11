import numpy as np
import itertools
import scipy
from plotly import graph_objects as go 

# verts = np.array(list(itertools.product([0, 1], repeat=3)))

verts = np.random.randn(100, 4)
verts /= np.linalg.norm(verts, axis=1, keepdims=True)


np.random.seed(42)
# Generate 100 random points on unit sphere
# verts = np.random.randn(1000, 4)
# verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)




j = scipy.spatial.Delaunay(verts)
T = j.simplices
V = j.points

# Extract all triangular faces from tetrahedra
all_faces = []
for simplex in T:
    # Each tet [a,b,c,d,e] = [0,1,2,3,4] has (5 choose 4) faces
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[3]]) #4 missing
    all_faces.append([simplex[0], simplex[2], simplex[3], simplex[4]]) #1 missing 
    all_faces.append([simplex[0], simplex[1], simplex[3], simplex[4]]) #2 missing
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[4]]) #3 missing
    all_faces.append([simplex[1], simplex[2], simplex[3], simplex[4]]) # 0 missing 
   

# Keep only boundary faces (appear exactly once)
from collections import Counter
face_tuples = [tuple(sorted(f)) for f in all_faces]
boundary = [f for f, count in Counter(face_tuples).items() if count == 1]
boundary_faces = np.array([list(f) for f in boundary])


np.save("Delaunay_4D_simplices.npy", arr = T)
np.save("Delaunay_4D_points", arr = V)
# Write to file
with open("Delaunay 100point sphere.txt", "w") as f:
    # Write vertices first
    for vertex in V:
        f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]}\n")
    # Write boundary faces
    for face in boundary_faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3] +1}\n")  # +1 because OBJ is 1-indexed

# Plot surface mesh
# fig = go.Figure(data=[
#     go.Mesh3d(
#         x=V[:, 0], y=V[:, 1], z=V[:, 2],
#         i=boundary_faces[:, 0],
#         j=boundary_faces[:, 1],
#         k=boundary_faces[:, 2],
#         color='cyan',
#         opacity=0.6,
#         flatshading=True
#     )
# ])

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[-1.5, 1.5], autorange=False),
#         yaxis=dict(range=[-1.5, 1.5], autorange=False),
#         zaxis=dict(range=[-1.5, 1.5], autorange=False),
#         aspectmode='cube'
#     ),
#     title="Delaunay Surface of Unit Cube"
# )

# fig.show()