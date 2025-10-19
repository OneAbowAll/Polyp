import numpy as np

def fit_plane(points):
    """
    Fit a plane to a set of 3D points using least squares, removing outliers using the 80th percentile.
    Args:
        points (np.ndarray): Nx3 array of 3D points (float).
    Returns:
        (normal, d): normal vector (unit), and offset d in plane equation ax + by + cz + d = 0
    """
    points = np.asarray(points)
    # Initial fit
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    d = -np.dot(normal, centroid)
    # Compute distances to plane
    distances = np.abs((points @ normal) + d) / np.linalg.norm(normal)
    # Use 80th percentile as inlier threshold
    thresh = np.percentile(distances,50)
    inliers = distances <= thresh
    points_inliers = points[inliers]
    # Refit plane with inliers
    centroid = points_inliers.mean(axis=0)
    centered = points_inliers - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    d = -np.dot(normal, centroid)
    return normal, d,thresh

def project_point_on_plane(p, normal, d):
    """
    Projects a 3D point p onto a plane defined by normal and d (ax + by + cz + d = 0).
    Args:
        p (np.ndarray): 3D point (shape: (3,))
        normal (np.ndarray): Plane normal (shape: (3,))
        d (float): Plane offset
    Returns:
        np.ndarray: Projected 3D point on the plane
    """
    normal = np.asarray(normal)
    p = np.asarray(p)
    normal_unit = normal / np.linalg.norm(normal)
    distance = (np.dot(normal_unit, p) + d)
    return p - distance * normal_unit, abs(distance)


    def median_direction(points, percentile=50):
        """
        Computes the median 3D direction from a list of 3D unit vectors, discarding outliers using the given percentile.
        Args:
            points (np.ndarray or list): Nx3 array or list of 3D direction vectors.
            percentile (float): Percentile to keep (default 50, i.e., median).
        Returns:
            np.ndarray: The robust median 3D direction (unit vector).
        """
        points = np.asarray(points)

        # Compute the centroid direction and normalize
        centroid = np.mean(points, axis=0)
        centroid /= np.linalg.norm(centroid)
        # Compute cosine of angles (dot product) between each point and centroid
        cos_angles = np.clip(points @ centroid, -1.0, 1.0)
        # Use angle (arccos) as distance
        angles = np.arccos(cos_angles)
        # Determine inlier threshold
        thresh = np.percentile(angles, percentile)
        inliers = points[angles <= thresh]
        # Compute mean direction of inliers and normalize
        median = np.mean(inliers, axis=0)
        median /= np.linalg.norm(median)
        # Pick the input direction closest to the median
        cos_angles = np.clip(points @ median, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        closest_idx = np.argmin(angles)
        return points[closest_idx] 
    