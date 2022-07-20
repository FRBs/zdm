"""
Generates figure A1 from the paper
uses data in directory A1
"""


import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

import matplotlib

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)


def make_histogram(oldx,oldy):
    """ makes cumulative x,y into something like a histogram """
    newx=np.zeros([oldx.size*2+2])
    newy=np.zeros([oldx.size*2+2])
    newx[0]=0.
    newx[-1]=100
    
    newy[-1]=0.
    newy[-2]=0.
    for i,x in enumerate(oldx):
        newx[2*i+1]=x
        newx[2*i+2]=x
        newy[2*i]=oldy[i]
        newy[2*i+1]=oldy[i]
        
    return newx,newy

data=np.loadtxt("FigureA1/frb_data.dat")

DM=data[:,2]
SNR=data[:,4]
FREQ=data[:,0]

############### plots cumulative data ###########

LOW=np.where(FREQ==1)[0]
MID=np.where(FREQ==2)[0]
HIGH=np.where(FREQ==3)[0]

SNR_LOW=SNR[LOW]

SNR_MID=SNR[MID]
SNR_HIGH=SNR[HIGH]
SNR=np.sort(SNR)
SNR_LOW=np.sort(SNR_LOW)
SNR_MID=np.sort(SNR_MID)
SNR_HIGH=np.sort(SNR_HIGH)
y=(np.arange(SNR.size)+1)[::-1]
yLOW=(np.arange(SNR_LOW.size)+1)[::-1]
yMID=(np.arange(SNR_MID.size)+1)[::-1]
yHIGH=(np.arange(SNR_HIGH.size)+1)[::-1]


def logtofit(x,a):
    return a-1.5*x

def tofit(x,a):
    return a*x**-1.5

p,pc=curve_fit(logtofit,np.log10(SNR),np.log10(y),p0=[100])
pLOW,pc=curve_fit(logtofit,np.log10(SNR_LOW),np.log10(yLOW),p0=[100])
pMID,pc=curve_fit(logtofit,np.log10(SNR_MID),np.log10(yMID),p0=[100])
pHIGH,pc=curve_fit(logtofit,np.log10(SNR_HIGH),np.log10(yHIGH),p0=[100])

SNR,y=make_histogram(SNR,y)
SNR_LOW,yLOW=make_histogram(SNR_LOW,yLOW)
SNR_MID,yMID=make_histogram(SNR_MID,yMID)
SNR_HIGH,yHIGH=make_histogram(SNR_HIGH,yHIGH)


plt.figure()
plt.ylabel('$N_{\\rm FRB} > {\\rm SNR}$')
plt.xlabel('${\\rm SNR}$')

plt.ylim(0.8,30)
plt.xlim(9,50)
plt.xscale('log')
plt.yscale('log')

fitx=np.linspace(9,100)
fity=fitx**-1.5
fity *= 700


plt.plot(SNR,y,label='CRAFT/ICS',linewidth=3)
plt.plot(fitx,tofit(fitx,10**p),color=plt.gca().lines[-1].get_color())
plt.plot(SNR_LOW,yLOW,label='CRAFT/ICS 900 MHz',linestyle='--',linewidth=3)
plt.plot(fitx,tofit(fitx,10**pLOW),color=plt.gca().lines[-1].get_color(),linestyle='--')
plt.plot(SNR_MID,yMID,label='CRAFT/ICS 1.2 GHz',linestyle='-.',linewidth=3)
plt.plot(fitx,tofit(fitx,10**pMID),color=plt.gca().lines[-1].get_color(),linestyle='-.')

#plt.plot(SNR_HIGH,yHIGH,label='CRAFT/ICS 1.6 GHz')

plt.xticks([10,20,30,40,50],['10','20','30','40','50'])
plt.yticks([1,10],['1','10'])

#plt.plot(fitx,fity/2,label='${\\rm SNR}^{-1.5}$')

plt.legend()
plt.tight_layout()

plt.savefig('FigureA1/logN_logS.pdf')
plt.close()
