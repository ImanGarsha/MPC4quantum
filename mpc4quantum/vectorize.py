import numpy as np
import qutip as qt
import itertools as it
import math

from .linearize import create_power_list


def discretize_homogeneous(A_cts_list, dt, order):
    """
    Construct an Euler discretization of bilinear dynamics to a specified order.
    Assumes homogenous in the control.
    Requires computing commutators like a Dyson series.

    :param A_cts_list: A list of operators that act on the state. The first entry is the drift H0,
                       the second entry is paired with u_1, and so on.
    :param dt: Discrete timestep.
    :param order: Order of dt to take the expansion.
    :return: The discretization operator which may rely on a lifted control basis if a higher-order discretization
             was used.
    """
    dim_x = A_cts_list[0].shape[0]
    id_op = np.identity(dim_x)
    dim_u = len(A_cts_list) - 1
    
    full_powers_list = create_power_list(order, dim_u)
    A_dst_list = [np.zeros((dim_x, dim_x), dtype=complex) for _ in range(len(full_powers_list))]

    A_dst_list[0] = id_op.astype(np.complex128)

    for k in range(1, order + 1):
        prefactor = (dt ** k) / math.factorial(k)
        
        for product_indices in it.product(range(len(A_cts_list)), repeat=k):
            
            # --- START OF THE FIX ---
            # If there's only one matrix, no dot product is needed.
            # Otherwise, use multi_dot for an efficient product of the sequence.
            if len(product_indices) == 1:
                term_matrix = A_cts_list[product_indices[0]]
            else:
                term_matrix = np.linalg.multi_dot([A_cts_list[i] for i in product_indices])
            # --- END OF THE FIX ---

            control_powers = np.zeros(dim_u, dtype=int)
            for index in product_indices:
                if index > 0:
                    control_powers[index - 1] += 1
            
            for i, p in enumerate(full_powers_list):
                if np.array_equal(p, control_powers):
                    A_dst_list[i] += prefactor * term_matrix
                    break
                    
    return np.hstack(A_dst_list)

def vectorize_me(H, measure_list):
    dim_m = len(measure_list)

    # Precompute structure constants of measurement basis
    structure_table = []
    for i, sigma_i in enumerate(measure_list):
        for j, sigma_j in enumerate(measure_list):
            for k, sigma_k in enumerate(measure_list):
                struct_const = 0. if i == j else (qt.commutator(sigma_i, sigma_j).dag() * sigma_k).tr()
                structure_table.append([i, j, k, struct_const])
    structure_table = np.array(structure_table).reshape(dim_m, dim_m, dim_m, -1)[:, :, :, -1]

    # Project hamiltonian
    H_list = [(H.dag() * sigma_i).tr() for sigma_i in measure_list]

    # Project Liouville equation
    A_op = []
    for k in range(dim_m):
        for j in range(dim_m):
            entry = 0
            for i in range(dim_m):
                entry = entry + H_list[i] * structure_table[i, k, j]
            A_op.append([j, k, -1j * entry])
    return np.array(A_op).reshape(dim_m, dim_m, -1)[:, :, -1]
