import numpy as np
import os
import math
from scipy.linalg import eigh 

simplex_path = os.path.join(os.path.dirname(__file__), "6D_TORUS_DELAUNAY_SIMPLICES.npy")
simplices = np.load(simplex_path) #turn the delaunay into a np.saveto file
point_path = os.path.join(os.path.dirname(__file__), "6D_TORUS_DELAUNAY_POINTS.npy")
points = np.load(point_path)


n = points.shape[0]
assert  n > 1
W_i_j = np.zeros((n, n))

M_list = np.zeros((n,n))

for simplex in simplices: 

    simplex = np.sort(simplex)



    n_simplex_points = points[simplex]
    v_0, v_1, v_2, v_3, v_4, v_5, v_6 = n_simplex_points
    Mega_volume = (1/math.factorial(6)) * np.abs(np.linalg.det(np.hstack(((v_1 - v_0).reshape(-1, 1),(v_2 - v_0).reshape(-1,1), (v_3 - v_0).reshape(-1,1), (v_4 - v_0).reshape(-1,1), (v_5 - v_0).reshape(-1,1), (v_6 - v_0).reshape(-1,1)))))   #i.e 1/6! mult by the det of the matrix 
    #Constructing the M matrix
    for simplex_ind in simplex: 
        M_list[simplex_ind][simplex_ind] += (Mega_volume) / (6 + 1) #i.e vol over  n + 1 


    #Constructing the W matrix
    for j in range(6 + 1):
        for i in range(j +1, 6 + 1):

            
            Inner_super_hedra = simplex[(simplex != simplex[i]) & (simplex != simplex[j])] #i.e the things that's separate from the two points that we're considering in this edge

            vertex_points = points[Inner_super_hedra] #the simplex in quesiton has 6 points so we're removing two so there are 4 in here
            t_1, t_2, t_3, t_4, t_5 = vertex_points
            e1, e2, e3, e4 = t_2 - t_1, t_3 - t_1, t_4 - t_1, t_5 - t_1


            Gram_matrix = np.array([[np.dot(e1, e1), np.dot(e1, e2), np.dot(e1, e3), np.dot(e1, e4)],
                                    [np.dot(e2, e1), np.dot(e2, e2), np.dot(e2, e3), np.dot(e2, e4)],
                                    [np.dot(e3, e1), np.dot(e3, e2), np.dot(e3, e3), np.dot(e3, e4)],
                                    [np.dot(e4, e1), np.dot(e4, e2), np.dot(e4, e3), np.dot(e4, e4)]])  #NEED TO CHECK THIS, I JUST GUESSED BASED OFF THE 2X2 MATRIX USED IN THE 4D LAPLACIAN

            Det_gram =  np.linalg.det(Gram_matrix)
            Area = (1/2) *  np.sqrt(np.abs(Det_gram) + 1e-6)  #The 1 factor is a half in the 4D laplacian. im pretty sure the half is useless but leaev this as a TODO just in case


            #this is the dihedral angle between the two tetrahedra that 
            #intersect along this above triangle. 


            #process (Imperatore hath estimated it from the 3D example): it's to find the outer two points in this 
            #2-tetraheda formation. Then find their runoffs towards the midpoint of the triangle
            #then we find the orthogonal complemeent to the plane that's defined by the triangle (i.e the 3 points on its edges)
            #then we do the regular dot product between these 2 vectors. 


            #find the vec from both the i and j which pierce the midpoint of the common triangle
            mid_point = (t_1 + t_2 + t_3 + t_4 + t_5) / 5
            i_run_off = mid_point - points[simplex[i]]
            j_run_off = mid_point - points[simplex[j]]

            #find the comploment of each of the vectors against the 2d space that's spanned by the triangle
            
            
            Temp = np.column_stack([e1, e2, e3, e4])
            proj_mat = Temp.T @ Temp
            alpha_1 = np.linalg.solve(proj_mat, Temp.T @ i_run_off)
            alpha_2 = np.linalg.solve(proj_mat, Temp.T @ j_run_off)
            i_proj = Temp @ alpha_1
            j_proj = Temp @ alpha_2
            i_run_off = i_run_off - i_proj
            j_run_off = j_run_off - j_proj


            #Simple dot product calculation

            n1 = np.linalg.norm(i_run_off)
            n2 = np.linalg.norm(j_run_off)
            angle = np.clip( np.dot(i_run_off, j_run_off) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(angle)
        
           
            cot = 1 / (np.tan(angle) + 1e-9)

            contribution = (1/(6 * (6-1))) * cot * Area
            
            W_i_j[simplex[i]][simplex[j]] += contribution
            W_i_j[simplex[j]][simplex[i]] += contribution



Degree_matrix = -1 * np.diag(np.sum(-W_i_j, axis=0))
L = Degree_matrix - W_i_j
L = 0.5 * (L + L.T)  # enforce symmetry




eigenvalues, eigenvectors = eigh(L , M_list) # eigh returns ascending
eigenvalues = np.clip(eigenvalues, 0.0, None)




eigenvals_path = os.path.join(os.path.dirname(__file__), "6D_TORUS_EIGENVALS.npy")
eigenvecs_path = os.path.join(os.path.dirname(__file__), "6D_TORUS_EIGENVECS.npy")
np.save(eigenvals_path, eigenvalues)
np.save(eigenvecs_path, eigenvectors) #GET RID OF T

print("Smallest eigenvalues:", eigenvalues[:3])
print("eigenvectors shape:", eigenvectors.shape)


            





