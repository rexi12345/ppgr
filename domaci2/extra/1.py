import numpy as np
from numpy.linalg import svd

np.set_printoptions(precision=5, suppress=True)

def opsti_polozaj(tacke, eps):
    x1, x2, x3, x4 = tacke[0], tacke[1], tacke[2], tacke[3]
    if np.min(np.abs([np.linalg.det([x1, x2, x3]),
                      np.linalg.det([x1, x2, x4]),
                      np.linalg.det([x1, x3, x4]),
                      np.linalg.det([x2, x3, x4])])) < eps:
        return False #nisu u opstem polozaju
    else:
        return True #jesu u opstem polozaju


# Zamena za scipy.linalg.null_space koji nece da se ucita u automatskom testeru
# Prekopirano sa stackoverflow
# link: https://stackoverflow.com/questions/49852455/how-to-find-the-null-space-of-a-matrix-in-python-using-numpy
def nullspace(A, atol=1e-13, rtol=0):
    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns

def zameni_nulu(row):
    for i, x in enumerate(row):
        if abs(x) < 1e-10:
            row[i] = 0

    return row

def pozitivne_nule(A):
    row1 = zameni_nulu(A[0])
    row2 = zameni_nulu(A[1])
    row3 = zameni_nulu(A[2])
    return np.asmatrix([row1, row2, row3])

def naivni(origs, imgs):
    origs = np.array(origs)
    imgs = np.array(imgs)
    eps = 0.00001
    x1, x2, x3, x4 = origs
    x1p, x2p, x3p, x4p = imgs

    if not opsti_polozaj(origs, eps):
        return "Losi originali!"
    if not opsti_polozaj(imgs, eps):
        return "Lose slike!"

    diag = nullspace(np.transpose([x1, x2, x3, x4]))[:, 0][:-1]
    f = np.transpose(np.diag(diag).dot([x1, x2, x3]))

    diag = nullspace(np.transpose([x1p, x2p, x3p, x4p]))[:, 0][:-1]
    g = np.transpose(np.diag(diag).dot([x1p, x2p, x3p]))

    h = g.dot(np.linalg.inv(f))
    h /= 2       #korigovanje
    h /= h[2][2] #normalizacija
    h = pozitivne_nule(h)

    return h


trapez = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1]]
pravougaonik = [[- 2, - 1, 1], [2, - 1, 1], [2, 1, 1], [- 2, 1, 1]]

origs = [[- 3, - 1, 1], [3, - 1, 1], [1, 1, 1], [- 1, 1, 1]]
imgs = [[- 2, - 5, 1], [2, - 5, 1], [2, 1, 1], [6, -3, 3]]   #primetite da nisu u opstem polozaju

print(naivni(trapez, pravougaonik))
print(naivni([[868, 2, 1],[410, 375, 1], [1278, 813, 3], [499, 222, 1]], [[567,934, 1], [394, 110, 1], [535, 777, 1], [169, 694, 1]]))
print(naivni(origs, imgs))