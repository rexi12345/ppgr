import numpy as np
import cv2
from numpy.linalg import svd

np.set_printoptions(precision=5, suppress=True)

#Pomocne funkcije

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


# Prikazivanje slike kutije
img_path = 'originalna_slika.jpg'
img = cv2.imread(img_path)
cv2.namedWindow('originalna_slika')

niz=[]
counter = 0

def click_event(event,x,y,f,param):
    global img, counter

    if event == cv2.EVENT_LBUTTONDOWN:
        niz.append([x, y, 1])
        counter += 1

        if counter <= 4:
            cv2.circle(img, (x, y), 1, (0, 255, 0), 2)
        else:
            cv2.circle(img, (x, y), 1, (255, 0, 0), 2)
        cv2.imshow('originalna_slika', img)

    # Kada smo izabrali 8 tacaka crta se nova slika
    if len(niz) == 8:
        origs = niz[0:4]
        imgs = niz[4:8]
        M = naivni(origs, imgs)
        weight, height, _ = img.shape

        img2 = cv2.warpPerspective(img,M,(500,889),cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        cv2.imshow('ispravljena',img2)


cv2.setMouseCallback('originalna_slika', click_event)
cv2.imshow('originalna_slika', img)
cv2.waitKey(0)
cv2.destroyAllWindows()