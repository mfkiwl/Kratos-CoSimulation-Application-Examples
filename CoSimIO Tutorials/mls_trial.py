import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def weight_function(xi, xj, h):
    """Gaussian weight function for MLS"""
    return np.exp(-np.linalg.norm(xi - xj) ** 2 / (h ** 2))

def calculate_h(known_points, x_test, k=5, scale_factor=1.5):
    """
    Estimate local h for Moving Least Squares (MLS) based on nearest neighbor distances.

    Parameters:
        known_points (array): List of (x, y) coordinates of known data points.
        x_test (tuple): Query point (x, y).
        k (int): Number of nearest neighbors to consider.
        scale_factor (float): Multiplier for the average distance.

    Returns:
        float: Estimated h value.
    """
    tree = KDTree(known_points)  # Build KDTree for efficient neighbor search
    distances, _ = tree.query(x_test, k+1)  # Find k nearest neighbors (excluding itself)
    
    avg_distance = np.mean(distances[1:])  # Compute mean neighbor distance (ignoring itself)
    h_estimated = scale_factor * avg_distance  # Scale for better smoothing

    return h_estimated


def quadratic_mls(x, y, values, x_query, h=1.0):
    """Quadratic Moving Least Squares interpolation at x_query"""
    # Construct the quadratic basis: [1, x, y, x^2, xy, y^2]
    def basis(x, y):
        return np.array([1, x, y, x**2, x*y, y**2])

    A = np.zeros((6, 6))
    b = np.zeros(6)
    
    for xi, yi, vi in zip(x, y, values):
        phi = basis(xi, yi)  # Basis functions
        w = weight_function(np.array([xi, yi]), np.array(x_query), h)  # Weight
        A += w * np.outer(phi, phi)
        b += w * vi * phi

    # Solve for coefficients
    coeffs = np.linalg.solve(A, b)
    
    # Compute the interpolated value at x_query
    return coeffs @ basis(x_query[0], x_query[1])

# Generate sample scattered data
np.random.seed(42)
x = np.random.rand(20) * 10
y = np.random.rand(20) * 10
known_points = np.vstack((x, y)).T  # Convert to (N,2) format
values = np.sin(x) + np.cos(y)  # Example function

x_test = (5.0, 5.0)

h_dynamic = calculate_h(known_points, x_test, k=3, scale_factor=1)
interpolated_value = quadratic_mls(x, y, values, [x_test[0], x_test[1]], h_dynamic)

print(interpolated_value)
print(np.sin(x_test[0]) + np.cos(x_test[1]))


