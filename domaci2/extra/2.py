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

trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1], [1,2,3], [-8,-2,1]]
pravougaonik1 = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1], [2,1,5], [-16,-5,5]]
print(DLT(trapez, pravougaonik1))