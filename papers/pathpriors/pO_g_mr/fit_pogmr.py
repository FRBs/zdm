"""

This script fits data to the p(U|mr) data, i.e. the probability that
a galaxy is unseen given it has an intrinsic magnitude of m_r.

The functional form of the fits is given by opt.pUgm

The data is provided by Michelle Woodland (for CRAFT)
and Bridget Anderson (data to be published)


"""



from zdm import optical as opt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


import matplotlib

defaultsize=14
ds=4
font = {'family' : 'Helvetica',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)



# relevant for CRAFT, from Bridget
df = pd.read_csv("pu_mr_vs_mag_CRAFT_VLT_FORS2_r.csv")
result = curve_fit(opt.pUgm,df['mag'],df['PU_mr'],p0=[26.5,0.3])
CRAFT_result = result[0]
CRAFT_pogm = opt.pUgm(df['mag'],result[0][0],result[0][1])

# Legacy surveys - from Bridget Anderson
Lmags = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
LpU_mr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027122579985493736,
        0.02436363700867187, 0.014231286256411943, 0.047649506708623335,
        0.15510269056554593, 0.4759090774425562, 0.8798642289140987, 1.0,
        0.980904733057884]
result = curve_fit(opt.pUgm,Lmags,LpU_mr,p0=[26.5,0.3])
Legacy_result = result[0]
Legacy_fit = opt.pUgm(Lmags,result[0][0],result[0][1])

#Pan-STARRS: from Bridget Anderson
PSmags = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
            21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0]
PSpU_mr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05338978533359865,
            0.0753794861936344, 0.17783737932531407, 0.6123903316329317,
            0.9170697731444298, 0.9736154053312236, 1.0, 0.9704756326566667,
            0.9993072645593615]
result = curve_fit(opt.pUgm,PSmags,PSpU_mr,p0=[26.5,0.3])
PS_result = result[0]
PS_fit = opt.pUgm(PSmags,result[0][0],result[0][1])

print("Fit results are...")
print("     PanSTARSS: ",PS_result)
print("     Legacy ",Legacy_result)
print("     VLT/FORS2 ",CRAFT_result)

LpU_mr = np.array(LpU_mr)
PSpU_mr = np.array(PSpU_mr)

plt.figure()

plt.plot(df['mag'],1.-df['PU_mr'],label="VLT/FORS2")
plt.plot(df['mag'],1.-CRAFT_pogm,label="         (fit)",linestyle="--",color = plt.gca().lines[-1].get_color())
plt.plot(Lmags,1.-LpU_mr,label="Legacy surveys")
plt.plot(Lmags,1.-Legacy_fit,label="         (fit)",linestyle="--",color = plt.gca().lines[-1].get_color())
plt.plot(PSmags,1.-PSpU_mr,label="Pan-STARRS")
plt.plot(PSmags,1.-PS_fit,label="         (fit)",linestyle="--",color = plt.gca().lines[-1].get_color())
plt.legend()
plt.xlim(15,30)
plt.ylim(0,1)
plt.xlabel("$m_r$")
plt.ylabel("$p(O|m_r)$")
plt.tight_layout()
plt.savefig("pOgm.png")
plt.close()

