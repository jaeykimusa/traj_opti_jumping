# math_utils.py

import numpy as np
from scipy.interpolate import interp1d
import casadi as ca

def getPseudoInverse(A):
    return np.linalg.pinv(A)

def getZerosMatrix(r, c):
    return np.zeros((r, c))

def getIdentityMatrix(i):
    return np.eye(i)

def printSize(matrix):
    print(matrix.shape)

def getRowSize(matrix):
    return matrix.shape[0]
def getColumnSize(matrix):
    return matrix.shape[1]

def interpolateMatrixToTargetColumns(original_matrix, target_columns):
    """
    Linearly interpolates each row of a 2D numpy array to a target number of columns.

    Args:
        original_matrix (np.ndarray): The input matrix with shape (num_rows, original_columns).
        target_columns (int): The desired number of columns for the interpolated matrix.

                              This must be greater than or equal to original_columns.

    Returns:
        np.ndarray: The interpolated matrix with shape (num_rows, target_columns).
    """
    num_rows, original_columns = original_matrix.shape

    if target_columns < original_columns:
        raise ValueError(
            "Target number of columns must be greater than or equal to original columns for upsampling."
        )

    # Create an array to store the interpolated matrix
    interpolated_matrix = np.zeros((num_rows, target_columns))

    # Define the original x-coordinates (representing the column indices)
    # These range from 0 to original_columns - 1
    original_x = np.arange(original_columns)

    # Define the new x-coordinates for interpolation
    # These will span the same range as original_x, but with more points.
    new_x = np.linspace(0, original_columns - 1, target_columns)

    # Iterate through each row and perform linear interpolation
    for i in range(num_rows):
        # Create a linear interpolation function for the current row
        interp_func = interp1d(original_x, original_matrix[i, :], kind='linear')

        # Evaluate the interpolation function at the new x-coordinates
        interpolated_matrix[i, :] = interp_func(new_x)

    return interpolated_matrix


def convert_3Drot_to_quat(q_partial):
    """
    Convert 3D config vector (with 3D rotation) to 4D quaternion vector for floating base.
    q_partial: [pos(3), rotvec(3), joint(n-6)] → (18,)
    return: [pos(3), quat(4), joint(n-6)] → (19,)
    Accepts CasADi symbolic or NumPy input.
    """
    assert q_partial.shape[0] == 18, "Expected 18-element vector"

    pos = q_partial[0:3]
    rotvec = q_partial[3:6]
    joint = q_partial[6:]

    # Handle CasADi symbolic types
    if isinstance(q_partial, (ca.SX, ca.MX)):
        angle = ca.norm_2(rotvec)
        axis = rotvec / (angle + 1e-8)
        quat = ca.vertcat(ca.cos(angle / 2), axis * ca.sin(angle / 2))
        q_full = ca.vertcat(pos, quat, joint)

    # Handle NumPy numeric input
    elif isinstance(q_partial, np.ndarray):
        angle = np.linalg.norm(rotvec)
        if angle < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = rotvec / angle
            quat = np.concatenate([[np.cos(angle / 2)], axis * np.sin(angle / 2)])
        q_full = np.concatenate([pos, quat, joint])

    else:
        raise TypeError(f"Unsupported type for q_partial: {type(q_partial)}")

    return q_full