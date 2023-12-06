import numpy as np
from numpy import linalg

np.set_printoptions(precision=5, suppress=True)

def two_equations(x, xp):
    x = np.array(x).flatten()
    xp = np.array(xp).flatten()

    return np.array([np.concatenate([[0, 0, 0], -xp[2] * x, xp[1] * x]),
                     np.concatenate([xp[2] * x, [0, 0, 0], -xp[0] * x])])

def DLT(origs, imgs):
    if len(origs) != len(imgs):
        return "Razlicit broj tacaka originala i slike!"
    AA = np.concatenate([two_equations(x, xp) for x, xp in zip(origs, imgs)], axis=0)
    _, _, VV = np.linalg.svd(AA)
    last = VV[-1, :]
    h = np.reshape(last, (3, 3))
    h *= -2 #korigovanje
    h /= h[2][2] #normalizacija

    return h

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


def DLTwithNormalization(origs, imgs):
    TT = normMatrix(origs)
    TTp = normMatrix(imgs)

    transformed_orig = [np.dot(TT, pt) for pt in origs]
    transformed_images = [np.dot(TTp, pt) for pt in imgs]

    projective_transform = DLT(transformed_orig, transformed_images)

    h = np.dot(np.linalg.inv(TTp), np.dot(projective_transform, TT))
    h /= h[2][2] #normalizacija

    return h


trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]]
pravougaonik1 = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1], [2,1,5], [-16,-5,5]]
print(DLTwithNormalization(trapez, pravougaonik1))