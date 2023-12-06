import numpy as np
from numpy import linalg

np.set_printoptions(precision=5, suppress=True)

def affinize(point):
    return np.array([point[0]/point[2], point[1]/point[2]])

def normMatrix(points):
    pts = np.array(points)

    afine = np.array([affinize(pt) for pt in pts])
    teziste = np.mean(afine, axis=0)
    afine = afine - teziste
    dist = np.mean(np.linalg.norm(afine, axis=1))

    norm_matrix = np.sqrt(2) / dist * np.eye(3)
    norm_matrix[0, 2] = -teziste[0]*norm_matrix[2][2]
    norm_matrix[1, 2] = -teziste[1]*norm_matrix[2][2]
    norm_matrix[2][2] = 1

    return norm_matrix


trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]]
print(normMatrix(trapez))
