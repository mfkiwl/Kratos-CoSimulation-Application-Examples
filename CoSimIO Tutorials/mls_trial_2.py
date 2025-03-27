import numpy as np
from scipy.spatial import KDTree

def quadratic_mls(x, y, values, x_query, k=5, scale_factor=1.5):
    """Quadratic Moving Least Squares (MLS) using only k-nearest neighbors"""

    def weight_function(xi, xj, h):
        """Gaussian weight function for MLS"""
        return np.exp(-np.linalg.norm(xi - xj) ** 2 / (h ** 2))

    def calculate_h(known_points, x_test, k=5, scale_factor=1.5):
        """Estimate local h based on nearest neighbor distances."""
        tree = KDTree(known_points)  # Build KDTree for efficient neighbor search
        distances, _ = tree.query(x_test, k+1)  # Get k nearest neighbors (excluding itself)
        avg_distance = np.mean(distances[1:])  # Ignore distance to itself
        return scale_factor * avg_distance  # Scale for better smoothing

    def basis(x, y):
        """Quadratic basis: [1, x, y, x^2, xy, y^2]"""
        return np.array([1, x, y, x**2, x*y, y**2])

    # Convert input data to (N,2) format for KDTree
    known_points = np.vstack((x, y)).T
    tree = KDTree(known_points)

    # Find k nearest neighbors
    distances, indices = tree.query(x_query, k=k)
    print(indices)
    x_k = x[indices]
    y_k = y[indices]
    values_k = values[indices]

    # Compute local h dynamically
    h_dynamic = calculate_h(known_points, x_query, k, scale_factor)

    # Initialize matrices
    A = np.zeros((6, 6))
    b = np.zeros(6)

    # Compute weighted least squares
    for xi, yi, vi in zip(x_k, y_k, values_k):
        phi = basis(xi, yi)
        w = weight_function(np.array([xi, yi]), np.array(x_query), h_dynamic)
        A += w * np.outer(phi, phi)
        b += w * vi * phi

    # Solve for coefficients
    coeffs = np.linalg.solve(A, b)

    # Compute the interpolated value at x_query
    return coeffs @ basis(x_query[0], x_query[1])

# ----------------- TEST EXAMPLE -----------------
# Generate sample scattered data
np.random.seed(42)
x = np.random.rand(100) * 10  # 100 random points
y = np.random.rand(100) * 10
values = np.sin(x) + np.cos(y)  # Example function

# Query point
x_test = (5.0, 5.0)

# Estimate interpolated value using only the k-nearest neighbors
interpolated_value = quadratic_mls(x, y, values, x_test, k=10, scale_factor=1.5)

# Compare with the real function value
real_value = np.sin(x_test[0]) + np.cos(x_test[1])

print(f"MLS Interpolated: {interpolated_value:.6f}")
print(f"Real Value: {real_value:.6f}")
print(f"Error: {abs(interpolated_value - real_value):.6f}")