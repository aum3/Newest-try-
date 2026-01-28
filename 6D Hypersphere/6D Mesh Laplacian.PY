import numpy as np
import os
import math
from scipy.linalg import eigh 

simplex_path = os.path.join(os.path.dirname(__file__), "6D_HYPERSPHERE_DELAUNAY_SIMPLICES.npy")
simplices = np.load(simplex_path) #turn the delaunay into a np.saveto file
point_path = os.path.join(os.path.dirname(__file__), "6D_HYPERSPHERE_DELAUNAY_POINTS.npy")
points = np.load(point_path)


n = points.shape[0]
assert  n > 1
W_i_j = np.zeros((n, n))

M_list = np.zeros((n,n))

for simplex in simplices: 

    simplex = np.sort(simplex)

    # Now simplex has 6 vertices (5D boundary facet)
    n_simplex_points = points[simplex]
    v_0, v_1, v_2, v_3, v_4, v_5 = n_simplex_points
    
    # Volume of 5D simplex: (1/5!) * sqrt(det(Gram)), with Gram = E^T E
    # E stacks the 5 edge vectors in the embedding space (6D), producing a 6x5 matrix.
    E = np.hstack([
        (v_1 - v_0).reshape(-1, 1),
        (v_2 - v_0).reshape(-1, 1),
        (v_3 - v_0).reshape(-1, 1),
        (v_4 - v_0).reshape(-1, 1),
        (v_5 - v_0).reshape(-1, 1),
    ])
    Gram_vol = E.T @ E  # 5x5 square matrix
    det_gram = np.linalg.det(Gram_vol)
    Mega_volume = (1 / math.factorial(5)) * np.sqrt(max(det_gram, 0.0) + 1e-12)
    
    # Constructing the M matrix
    for simplex_ind in simplex: 
        M_list[simplex_ind][simplex_ind] += (Mega_volume) / 6  # vol over n + 1 (where n=5)

    # Constructing the W matrix
    for j in range(6):
        for i in range(j + 1, 6):

            # The opposite 4-face (remove vertices i and j)
            Inner_super_hedra = simplex[(simplex != simplex[i]) & (simplex != simplex[j])]  # 4 vertices

            vertex_points = points[Inner_super_hedra]
            t_1, t_2, t_3, t_4 = vertex_points
            e1, e2, e3, e4 = t_2 - t_1, t_3 - t_1, t_4 - t_1, _5 - t_1

            Gram_matrix = np.array([[np.dot(e1, e1), np.dot(e1, e2), np.dot(e1, e3), np.dot(e1, e4)],
                                    [np.dot(e2, e1), np.dot(e2, e2), np.dot(e2, e3), np.dot(e2, e4)],
                                    [np.dot(e3, e1), np.dot(e3, e2), np.dot(e3, e3), np.dot(e3, e4)],
                                    [np.dot(e4, e1), np.dot(e4, e2), np.dot(e4, e3), np.dot(e4, e4)]])

            Det_gram = np.linalg.det(Gram_matrix)
            Area = (1/2) * np.sqrt(np.abs(Det_gram) + 1e-6)

            # Find the dihedral angle between the two facets sharing this 4-face
            mid_point = (t_1 + t_2 + t_3 + t_4) / 4
            i_run_off = mid_point - points[simplex[i]]
            j_run_off = mid_point - points[simplex[j]]

            # Project out the 4D space spanned by the opposite face
            Temp = np.column_stack([e1, e2, e3, e4])
            proj_mat = Temp.T @ Temp
            alpha_1 = np.linalg.solve(proj_mat, Temp.T @ i_run_off)
            alpha_2 = np.linalg.solve(proj_mat, Temp.T @ j_run_off)
            i_proj = Temp @ alpha_1
            j_proj = Temp @ alpha_2
            i_run_off = i_run_off - i_proj
            j_run_off = j_run_off - j_proj

            # Dihedral angle
            n1 = np.linalg.norm(i_run_off)
            n2 = np.linalg.norm(j_run_off)
            angle = np.clip(np.dot(i_run_off, j_run_off) / (n1 * n2 + 1e-9), -1.0, 1.0)
            angle = np.arccos(angle)
        
            cot = 1 / (np.tan(angle) + 1e-9)

            contribution = (1/(5 * 4)) * cot * Area  # (1/(n*(n-1))) where n=5
            
            W_i_j[simplex[i]][simplex[j]] += contribution
            W_i_j[simplex[j]][simplex[i]] += contribution
            # Simple dot product calculation

            n1 = np.linalg.norm(i_run_off)
            n2 = np.linalg.norm(j_run_off)
            angle = np.clip( np.dot(i_run_off, j_run_off) / (n1 * n2 + 1e-9), -1.0, 1.0)
            angle = np.arccos(angle)
        
            cot = 1 / (np.tan(angle) + 1e-9)

            contribution = (1/(5 * 4)) * cot * Area
            
            W_i_j[simplex[i]][simplex[j]] += contribution
            W_i_j[simplex[j]][simplex[i]] += contribution



Degree_matrix = -1 * np.diag(np.sum(-W_i_j, axis=0))
L = Degree_matrix - W_i_j
L = 0.5 * (L + L.T)  # enforce symmetry




eigenvalues, eigenvectors = eigh(L , M_list) # eigh returns ascending
eigenvalues = np.clip(eigenvalues, 0.0, None)




eigenvals_path = os.path.join(os.path.dirname(__file__), "6D_HYPERSPHERE_EIGENVALS.npy")
eigenvecs_path = os.path.join(os.path.dirname(__file__), "6D_HYPERSPHERE_EIGENVECS.npy")
np.save(eigenvals_path, eigenvalues)
np.save(eigenvecs_path, eigenvectors) #GET RID OF T

print("Smallest eigenvalues:", eigenvalues[:3])
print("eigenvectors shape:", eigenvectors.shape)


            





