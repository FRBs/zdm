
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)




# hard-coded color-color data

# array of g and I bands
gI = [ [21.23,19.875],
        [22.24,21.17],
        [22.59,21.10],
        [18.167,17.097],
        [24.02,22.41],
        [23.87,22.4],
        [21.037,19.618],
        [19.103,17.743],
        [23.3,21.90],
        [21.856,20.61],
        [20.910,19.564],
        [23.86,22.68],
        [20.476,19.194],
        [18.128,16.476],
        [15.819,14.860],
        [17.184,16.212],
        [21.49,20.47],
        [18.529,17.232]]

# hard-coded g minus R colours
gR = [[24.02,23.03],[20.842,20.258],[24.22,23.72],[18.529,17.843]]

RI = [[23.03,22.41],[17.843,17.232]]

# convert to numpy
gI = np.array(gI)
gR = np.array(gR)
RI = np.array(RI)


plt.xlabel("$g-R$")
bins = np.linspace(0,2,21)



print("Mean g minus I is ",np.mean(gI[:,0]-gI[:,1]),gI[:,1].size)
print("Mean R minus I is ",np.mean(RI[:,0]-RI[:,1]),RI[:,1].size)
print("Mean g minus R is ",np.mean(gR[:,0]-gR[:,1]),gR[:,1].size)
plt.figure()
plt.xlim(0.8,1.8)
plt.yticks(np.linspace(0,4,5))
plt.hist(gI[:,0]-gI[:,1],bins=bins,label="$m_g-m_I$",alpha=0.5)
plt.hist(2.*(gR[:,0]-gR[:,1]),bins=bins,label="$2(m_g-m_R)$",alpha=0.5)
plt.hist(2.*(RI[:,0]-RI[:,1]),bins=bins,label="$2(m_R-m_I)$",alpha=0.5)
plt.legend(loc = "upper left")
plt.xlabel("colour")
plt.ylabel("counts")
plt.tight_layout()
plt.savefig("color_correction.png")
plt.close()


