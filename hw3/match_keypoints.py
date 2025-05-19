import numpy as np

class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    dists = np.linalg.norm(des1[:, np.newaxis] - des2[np.newaxis, :], axis=2)
    best_in_des2 = np.argmin(dists, axis=1) 
    best_in_des1 = np.argmin(dists, axis=0)
    matches = []
    for i, j in enumerate(best_in_des2):
        if best_in_des1[j] == i:
            distance = dists[i, j]
            matches.append(DummyMatch(i, j, distance))

    matches.sort(key=lambda match: match.distance)
    
    return matches