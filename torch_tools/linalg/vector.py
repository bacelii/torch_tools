"""
Purpose: utility functions and wrappers
for working with vectors

"""
import numpy as np

def magnitude(vector):
    return np.linalg.norm(vector)

magn = magnitude
mag = magnitude
norm = magnitude

def unit(vector):
    return vector/magnitude(vector)

unit_vector = unit


def projection(v1,v2,return_unit = False):
    proj =  np.dot(v1,v2)/np.dot(v1,v1)*v1
    if return_unit:
        proj = unit(proj)
    return proj

def projection_error(v1,v2,return_unit = True):
    error  = v2 - projection(v1,v2)
    if return_unit:
        error = unit(error)
    return error

def cos_from_vectors(v1,v2,verbose = False):
    cos= np.clip(np.dot(unit(v1),unit(v2)),-1,1)
    if verbose:
        print(f"cos = {cos}")
    return cos

def theta_from_vectors(v1,v2,verbose = False,radians = True):
    theta =  np.arccos(cos_from_vectors(v1,v2))
    if not radians:
        theta = 180/np.pi
    if verbose:
        print(f"theta = {theta}")
    
    return theta

def angle_between_vectors(v1,v2,verbose = False):
    return theta_from_vectors(v1,v2,verbose =verbose,radians = False)

def vector_sum(v1,v2):
    return v1 + v2
def vector_sum_proj_strength(v1,v2):
    return np.sum(vector_sum(v1,v2))
def vector_sum_magnitude(v1,v2):
    return np.linalg.norm(vector_sum(v1,v2))

def magnitude_from_linear_equation(v1,v2,c1,c2):
    a = vec.magnitude(v1)
    b = vec.magnitude(v2)
    return np.sqrt(c1**2*a**2 + c2**2*b**2 + 2*c1*c2*np.dot(v1,v2))

def vector_from_theta_projection_simplified(v1,v2,theta,verbose = False):
    """
    Purpose: Given 2 vectors that define a plane, get the resultant vector that
    results from rotating v1 by angle theta in this plane
    (where positive theta in closest direction to v2)
    """
    v1_mag = np.linalg.norm(v1)
    return np.cos(theta)*v1 + v1_mag*np.sin(theta)*vec.projection_error(v1,v2,return_unit = True)


def intuitive_dot_product_computation_from_diff(
    v1,
    v2,
    verbose = True,):
    """
    You can compute the dot product in a certain way that gives you 
    good intuition about why the number came out that way
    by decoupling the magnitude and orientation in the following manner.
    
    Why works? By constraining the vectors to be unit vectors,
    the cosine is just the length of projection, and can solve easily 
    for it using the law of cosines or pythagorean theorm
    
    c = |x - y|
    
    c**2 = sin**2 + (1 - cos)
    c**2 = sin**2 + 1 + cos**2 - 2*cos
    c**2 = 1 + 1 - 2*cos
    c**2 = 2 - 2*cos
    cos = 1 - 0.5*|x - y|**2
    cos = 1 - o.5*sum((xi - yi)**2)

    Process: 
    With vectors v,g and elements vn and gn
    1) Convert both vectors to unit_vector * scale
        c1*vh, c2*gh  where vh,gh are unit vectors
    2) Find the difference of the 2 unit vectors (vh - gh) = D
    3) Square each element in the difference vector
        E = (vhn - ghn)**2
    4) The dot product of vh and gh is just the cosine because they are unit vectors.
    And the dot product can be computed in the following manner

        dot product = 1 - 0.5*sum(En)
        cos(theta)  = 1 - 0.5*sum(En)

    Summary of Process: For each element of the difference vector of the unit vectors,
    if you square and divide by 2,
    that is the amount you subtract from the cosine of the angle between them

    What does this show intuitively?
    - the cosine is maximized when there is no disagreement between unit vectors,
    and the more disagreement in unit vectors, the smaller the cosine, and the 
    wider the angle between 2 vectors in their enclosing 2D plane

    Why does this interpretation work?
    - by constraining the vectors to be unit vectors, the difference vector
    of the vectors contains all the information needed for computing the cosine
    of the angle

    How to relate to the dot product:

    Maximum Magnitude**2 increase of sum of vectors when they are in same orientation
        (increase in refernce to orthogonal orientation) = |v|*|g|
    Term describing the effect of deviating from same orientation = cos(theta) = 1 - 0.5*sum(En)
        -> Acts like a scaling term for the maximum

        Dot product = |v|*|g|*cos(theta) = [v]*[g]*(1 - 0.5*sum(En))
        Dot product = |v|*|g|*cos(theta) = [v]*[g]*(1 - 0.5*sum((vhn - ghn)**2))

    Conclusion: Once decouple the magnitude from the scale, just need to 
    conceptualize the difference unit vector, sum the squared elements, transform to get the cosine
    and then scale by the original magnitudes
    -> fits with the overall theme of transforming the problem into an easier problem
        and then transforming back afterwards

    """
    v1_mag = vec.norm(v1)
    v2_mag = vec.norm(v2)

    v1h = v1/v1_mag
    v2h = v2/v2_mag

    d = v1h - v2h   
    e = d**2

    cos_theta = 1 - 0.5*np.sum(e)

    dot_product = v1_mag*v2_mag*cos_theta

    if verbose:
        print(f"d = {d}")
        print(f"e = {e}")
        print(f"cos(theta) = {cos_theta}")
        print(f"intuitive manner dot_product= {dot_product}")
        print(f"Using standard dot_product = {np.dot(v1,v2)}")

    return dot_product

def example_intuitive_dot_product_computation_from_diff():
    v = np.array([1,3,5,7])
    g = np.array([1,10,5,7]) + 1
    intuitive_dot_product_computation_from_diff(
        v,
        g,
        verbose = True
    )  
    
def intuitive_dot_product_computation_from_product(
    v,
    g,
    verbose = True):
    """
    If we want to look at how the hadamard product (elementwise multiplicaiton)
    gives us the same outcome as the intuitive dot product (using the difference vector)
    we just expand our intuitive difference vector equation a little

    given unit vectors vh and gh, dh = vh - gh


    cos = 1 - 0.5*sum(dhi**2)
    cos = 1 - 0.5*sum((vhi - ghi)**2)
    #expanding out the square
    cos = 1 - 0.5 * sum( vhi**2 + ghi**2 - 2*vhi*ghi)
    cos = 1 - sum( [vhi**2 + ghi**2]/2  - vhi*ghi)

    The term [vhi**2 + ghi**2]/2  - vhi*ghi will always be greater than zero
        because it is equal to dhi**2

    --- alternate proof for this step ---
    real numbers a and b, call b = a + c

    a**2 + b**2 - 2*a*b > 0
    a**2 + (a+c)**2 - 2*a*(a + c) > 0
    a**2 + a**2 + c**2 + 2*a*c - 2*a**2 - 2*a*c > 0
    # everything cancels out other than c**2
    c**2 > 0
    -------------------------------------------

    Interpretation: 
    - [vhi**2 + ghi**2]/2  --> average magn squared of all projections
    - vhi*ghi --> product of each projection
    - define e (error) as the difference between average proj magn squared and 
    the product of the projection

        ei = [vhi**2 + ghi**2]/2 - vhi*ghi # will always be positive

    Rule: everytime this error term is non-zero, it indicates that
    the orientation between vectors does not match, and subtracts from 
    the ideal cosine value of 1 (when vectors have same orientation)

    Scenarios: 
    1) if cos = 1 (unit vectors match orientation), ei = 0
    2) The larger the orientation difference 
        --> the larger the sum(ei)
            --> the smaller the cosine of the angle

    Why care about intuition: 
    - when computing the dot product by hand, the 
    value of each element (a*b) should be compared 
    in reference to (a**2 + b**2)/2


    Process: 
    1) Separate the magnitude from the vector orientations
    2) Compute the average magnitude squared of unit vectors
    3) Compute the hadamard product
    4) Find error between average magn squared and hadamard
    5) Sum all errors
    6) Subtract from one to find cosine
    7) Multiply the magnitudes back
    """
    vh = vec.unit(v)
    gh = vec.unit(g)

    v_magn  = vec.magn(v)
    g_magn = vec.magn(g)

    avg_mag_sq = (vh**2 + gh**2 )/2
    hadamard_prod = vh*gh
    mag_sq_diff = avg_mag_sq - hadamard_prod

    cosine_manually = 1 - np.sum(mag_sq_diff)
    dot_product_manually = v_magn*g_magn*cosine_manually

    if verbose:
        print(f"cosine_manually = {cosine_manually}")
        print(f"dot_product_manually = {dot_product_manually}")
        print(f"dot_product_numpy = {np.dot(v,g)}")
        
    return dot_product_manually

def example_intuitive_dot_product_computation_from_product():
    v = np.array([1,3,5,7])
    g = np.array([1,10,5,7]) + 1
    
    return intuitive_dot_product_computation_from_product(
        v,g,verbose = True
    )

    
import vector as vec