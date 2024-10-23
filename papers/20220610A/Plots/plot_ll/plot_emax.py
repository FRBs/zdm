"""
This script plots the log likelihood for Emax,
taken from the datafile FRB20220610A_Emax.csv
as used in Ryder et al.

"""

import numpy as np
from matplotlib import pyplot as plt



import matplotlib
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

infile='FRB20220610A_Emax.csv'
data=np.genfromtxt(infile,delimiter=',')

print(data.shape)

lls=data[:,-9]
Emaxs=data[:,-1]

### creates probability distribution ####

lls2 = lls - np.max(lls)
pEmax = 10**lls2

plt.figure()


l1,=plt.plot(Emaxs,lls,color='blue',linestyle='-',marker='o',markerfacecolor='green',
    markeredgecolor='green',label="$L(E_{\\rm max})$")

plt.xlabel('$\\log_{10} (E_{\\rm max} / {\\rm erg})$')
plt.ylabel(r'$\log_{10} L(E_{\rm max})$')

ax=plt.gca()

plt.yticks((-420,-410,-400,-390))

ax2=ax.twinx()
ax2.set_ylabel('$p(E_{\\rm max})$')
l2,=ax2.plot(Emaxs,pEmax,color='red',linestyle='--',linewidth=2,label='$p(E_{\\rm max})$')
plt.ylim(0,1)

plt.legend(handles=(l1,l2),labels=("$L(E_{\\rm max})$",'$p(E_{\\rm max})$'),loc='lower right')
plt.tight_layout()
plt.savefig('emaxll.pdf')
plt.close()
