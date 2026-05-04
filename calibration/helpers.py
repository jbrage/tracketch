import numpy as np


def compute_closest_distances(
    r_exp: np.ndarray,
    z_exp: np.ndarray,
    r_contour: np.ndarray,
    z_contour: np.ndarray,
) -> np.ndarray:
    """
    Compute minimum distance from each experimental point to the contour line.

    Uses point-to-segment distance for accuracy.

    Parameters
    ----------
    r_exp, z_exp : np.ndarray
        Experimental point coordinates
    r_contour, z_contour : np.ndarray
        Contour line coordinates (ordered points)

    Returns
    -------
    np.ndarray
        Minimum distance from each experimental point to the contour
    """
    distances = np.zeros(len(r_exp))

    for i, (r_pt, z_pt) in enumerate(zip(r_exp, z_exp)):
        min_dist = np.inf

        # Check distance to each line segment
        for j in range(len(r_contour) - 1):
            # Segment endpoints
            r1, z1 = r_contour[j], z_contour[j]
            r2, z2 = r_contour[j + 1], z_contour[j + 1]

            # Vector from p1 to p2
            dr = r2 - r1
            dz = z2 - z1
            seg_len_sq = dr * dr + dz * dz

            if seg_len_sq < 1e-12:
                # Degenerate segment, use point distance
                dist = np.sqrt((r_pt - r1) ** 2 + (z_pt - z1) ** 2)
            else:
                # Project point onto line, clamp to segment
                t = max(0, min(1, ((r_pt - r1) * dr + (z_pt - z1) * dz) / seg_len_sq))

                # Closest point on segment
                r_closest = r1 + t * dr
                z_closest = z1 + t * dz

                dist = np.sqrt((r_pt - r_closest) ** 2 + (z_pt - z_closest) ** 2)

            min_dist = min(min_dist, dist)

        distances[i] = min_dist

    return distances
