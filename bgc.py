import numpy as np


def bgc_decompose(u):
    """
    Perform Balanced Group Commutator decomposition (BGCDecompose).
    
    Args:
        u: Input 2x2 unitary matrix.
        
    Returns:
        v, w: Two unitary matrices such that u ≈ vw*v†w†.
    """
    x = mat_to_cart3(u)
    n = np.linalg.norm(x)
    a = np.array([n, 0, 0])
    xu = cart3_to_mat(a)
    s = similarity_matrix(u, xu)
    as_conj = np.conj(s.T)
    vs, ws = x_group_factor(xu)
    v = s @ vs @ as_conj
    w = s @ ws @ as_conj
    return v, w


def mat_to_cart3(u):
    """
    Converts a 2x2 unitary matrix to a 3D Cartesian vector.
    """
    sx1 = -1 * np.imag(u[0, 1])
    sx2 = np.real(u[1, 0])
    sx3 = np.imag(u[1, 1])
    costh = 0.5 * np.real(u[0, 0] + u[1, 1])
    sinth = np.sqrt(sx1**2 + sx2**2 + sx3**2)
    
    x = np.zeros(3)
    if sinth == 0:
        x[0] = 2 * np.arccos(costh)
    else:
        th = np.arctan2(sinth, costh)
        x[0] = 2 * th * sx1 / sinth
        x[1] = 2 * th * sx2 / sinth
        x[2] = 2 * th * sx3 / sinth
    
    return x


def cart3_to_mat(x):
    """
    Converts a 3D Cartesian vector to a 2x2 unitary matrix.
    """
    th = np.sqrt(np.dot(x, x))
    if th == 0:
        return np.eye(2)
    else:
        return (
            np.cos(th / 2) * np.eye(2) 
            - 1j * np.sin(th / 2) / th * (
                x[0] * np.array([[0, 1], [1, 0]]) + 
                x[1] * np.array([[0, -1j], [1j, 0]]) + 
                x[2] * np.array([[1, 0], [0, -1]])
            )
        )


def similarity_matrix(u, xu):
    """
    Finds the similarity matrix for transforming u to xu.
    """
    a = mat_to_cart3(u)
    b = mat_to_cart3(xu)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    ab = np.dot(a, b)
    s = np.cross(b, a)
    ns = np.linalg.norm(s)
    
    # Check if cross product is small
    if abs(ns) < 1e-10:  # Replace 1e-10 with a small constant (similar to constants.RE)
        return np.eye(2)
    else:
        s = (np.arccos(ab / (na * nb)) / ns) * s
        return cart3_to_mat(s)


def x_group_factor(xu):
    """
    Factorizes a unitary matrix xu as part of the group commutator decomposition.
    """
    a = mat_to_cart3(xu)
    st = np.power(0.5 - 0.5 * a[0], 0.25)
    ct = np.sqrt(1 - st**2)
    theta = 2 * np.arcsin(st)
    alpha = np.arctan(st)
    
    b = np.zeros(3)
    c = np.zeros(3)
    
    b[0] = theta * st * np.cos(alpha)
    c[0] = b[0]
    b[1] = theta * st * np.sin(alpha)
    c[1] = b[1]
    b[2] = theta * ct
    c[2] = -b[2]
    
    B = cart3_to_mat(c)
    W = cart3_to_mat(b)
    aB = np.conj(B.T)
    V = B
    W = similarity_matrix(W, aB)
    return V, W
