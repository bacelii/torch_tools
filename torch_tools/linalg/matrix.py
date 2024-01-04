"""
Utility functions and demos for matrix
operations
"""
import numpy as np

# -- array and vector to use for demos
M_default = np.array([
    [1,4,6,9],
    [10,20,-5,6],
    [3,8,9, 10],
])

x_default = np.array([7,8,9,10])

def matrix_vector_product(M,x):
    prod = M @ x
    return prod

def matrix_vector_product_magn(M,x):
    return np.linalg.norm(M @ x)

def intuitive_matrix_product_magn_from_individual_and_cross_terms(
    M=M_default,
    x=x_default,
    verbose = True,
    ):
    """
    Purpose: To demonstrate the matrix tranformation of a vector (product of matrix and vector)
    has a magnitude squred that is a sum of:
    1) scaled individual magnitudes of the matrix column vectors
        sum(xi**2 * ||mi||**2)
    2) scaled cross terms (dot products of column vectors)
        sum( 2 * xi * xj * dot(mi,mj) )
    """
    M #= np.array([[1,4],[2,5]])
    x #= np.array([4,6])

    scaled_indivudal_magn = [xi**2 * np.linalg.norm(mi)**2 
                            for xi,mi in zip(x,M.T)]

    n_columns = M.shape[1]

    cross_terms = []
    for i in range(n_columns):
        for j in range(n_columns):
            if j >= i:
                continue
            cross_term = 2*x[i]*x[j]*np.dot(M[:,i],M[:,j])
            cross_terms.append(cross_term)

    magn_squared = np.sum(scaled_indivudal_magn) + np.sum(cross_terms)
    magn = np.sqrt(magn_squared)    

    if verbose:
        magn_check = np.linalg.norm(M@x)
        print(f"magn = {magn}")
        print(f"magn_check = {magn_check}")

    return magn
