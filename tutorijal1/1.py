import numpy as np

p1=(682,633,1)
p2=(354,972,1)
p3=(141,844,1)
p5=(759,337,1)
p6=(312,670,1)
p7=(17,540,1)
p8=(497,286,1)

def nevidljiva(p1,p2,p3,p5,p6,p7,p8):
    xb = np.cross(np.cross(p2,p6), np.cross(p1,p5))
    yb = np.cross(np.cross(p5,p6), np.cross(p7,p8))
    p4 = np.cross(np.cross(p8,xb), np.cross(p3,yb))
    return p4

p4 = nevidljiva(p1,p2,p3,p5,p6,p7,p8)
l = []
for p in p4: 
    p /= p4[2]
    l.append(int(p))

print(l)



    

