import numpy as np
import itertools
import scipy
from plotly import graph_objects as go 
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

verts = np.load(script_dir / "6D_HYPERSPHERE_POINT_CLOUD.npy")

j = scipy.spatial.Delaunay(verts)
T = j.simplices
V = j.points

# Extract all triangular faces from tetrahedra
all_faces = []
for simplex in T:
    # simplex has 7 vertices: simplex[0] ... simplex[6]
    all_faces.append([simplex[1], simplex[2], simplex[3], simplex[4], simplex[5], simplex[6]])  # omitted: 0
    all_faces.append([simplex[0], simplex[2], simplex[3], simplex[4], simplex[5], simplex[6]])  # omitted: 1
    all_faces.append([simplex[0], simplex[1], simplex[3], simplex[4], simplex[5], simplex[6]])  # omitted: 2
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[4], simplex[5], simplex[6]])  # omitted: 3
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[3], simplex[5], simplex[6]])  # omitted: 4
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[3], simplex[4], simplex[6]])  # omitted: 5
    all_faces.append([simplex[0], simplex[1], simplex[2], simplex[3], simplex[4], simplex[5]])  # omitted: 6   

# Keep only boundary faces (appear exactly once)
from collections import Counter
face_tuples = [tuple(sorted(f)) for f in all_faces]
boundary = [f for f, count in Counter(face_tuples).items() if count == 1]
boundary_faces = np.array([list(f) for f in boundary])


np.save(script_dir / "6D_HYPERSPHERE_DELAUNAY_SIMPLICES.npy", arr = boundary_faces)
np.save(script_dir / "6D_HYPERSPHERE_DELAUNAY_POINTS.npy", arr = V)


with open(script_dir / "6D_HYPERSPHERE_OBJ_FILE.obj", "w") as f:
        # Write vertices first
    for vertex in V:
        f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n")
    # Write boundary faces
    for face in boundary_faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1} {face[3] +1} {face[4] +1} {face[5] +1}\n")  # +1 because OBJ is 1-indexed
