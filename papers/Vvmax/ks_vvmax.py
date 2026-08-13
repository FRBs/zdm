"""
Function to perform test of uniformity in V/Vmax

It analyses the V/Vmax distribuiton for alll values of
N_sfr and alpha, and performs a ks-test on each

it also produced publication plots showing the ks p-value
as a function of n_sfr, and the mean values of V/Vmax

"""
import numpy as np
from scipy.stats import kstest
from matplotlib import pyplot as plt
import matplotlib
import os

matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'Calibri',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(load=False):
    """
    Main outer loop
    """
    indir = "LocOutput/"
    histdir="Hists_VVmax/"
    ksdir="KS_VVmax/"
    sfrdir="SFR_test/"
    
    
    loc=True
    prefix="localised_"
    
    NSFR=21
    SFRs = np.linspace(0,2,NSFR)
    all_ks = []
    all_vbars = []
    tosave = np.zeros([4,NSFR])
    tosaveVbar = np.zeros([4,NSFR])
    all_labels = []
    for i,alpha in enumerate([0,-1.5]):
        if load:
            break
        kstats=np.zeros([NSFR])
        vbars=np.zeros([NSFR])
        for j,Nsfr in enumerate(SFRs):
            infile = prefix+"vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+".dat"
            histname = histdir + "hist_"+infile[:-4] + ".pdf"
            ksname = ksdir + "ks_" +infile[:-4] + ".pdf"
            p,vbar=do_ks_stats(indir+infile,histname,ksname)
            kstats[j]=p
            vbars[j] = vbar
        
        title=sfrdir+prefix+"alpha_"+str(alpha)+"_ks_results.pdf"
        #plot_ks_dist(SFRs,kstats,plotname)
        all_ks.append(kstats)
        all_vbars.append(vbars)
        all_labels.append(prefix+str(alpha))
        tosave[i,:]=kstats
        tosaveVbar[i,:]=vbars
        
    loc=False
    indir="zMacquartOutput/"
    prefix="v2macquart_"
    for i,alpha in enumerate([0,-1.5]):
        if load:
            break
        kstats=np.zeros([NSFR])
        vbars=np.zeros([NSFR])
        for j,Nsfr in enumerate(SFRs):
            infile = prefix+"vvmax_data_NSFR_"+str(Nsfr)[0:3]+"_alpha_"+str(alpha)+".dat"
            
            histname = histdir + "hist_"+infile[:-4] + ".pdf"
            ksname = ksdir + "ks_" +infile[:-4] + ".pdf"
            p,vbar=do_ks_stats(indir+infile,histname,ksname)
            kstats[j]=p
            vbars[j] = vbar
            
        title=sfrdir+prefix+"alpha_"+str(alpha)+"_ks_results.pdf"
        #plot_k_dist(SFRs,kstats,plotname)
        all_ks.append(kstats)
        all_labels.append(prefix+str(alpha))
        tosave[i+2,:]=kstats
        all_vbars.append(vbars)
        tosaveVbar[i+2,:]=vbars
    
    all_labels=[
            "$z_{\\rm loc}, \\alpha=0$",
            "$z_{\\rm loc}, \\alpha=-1.5$",
            "$z_{\\rm DM}, \\alpha=0$",
            "$z_{\\rm DM}, \\alpha=-1.5$"
            ]
    
    if load:
        tosave = np.load(sfrdir+"all_ks_stats.npy")
        all_ks=[]
        for i in np.arange(4):
            all_ks.append(tosave[i,:])
        
        tosaveVbar = np.load(sfrdir+"all_Vbars.npy")
        all_vbars=[]
        for i in np.arange(4):
            all_vbars.append(tosaveVbar[i,:])
    else:
        np.save(sfrdir+"all_ks_stats.npy",tosave)
        np.save(sfrdir+"all_Vbars.npy",tosaveVbar)
    
    plotname = sfrdir+"all_ks_results.pdf"
    plot_ks_dist(SFRs,all_ks,all_labels,plotname)
    plotname = sfrdir+"all_vbar_results.pdf"
    plot_vbar_dist(SFRs,all_vbars,all_labels,plotname)
    
def plot_ks_dist(sfrs,all_kp,labels,outfile):
    """
    Plots the distribution of ks p-valuesas a function of alphs
    """
    plt.figure()
    linestyles=["-","--","-.",":"]
    for i,kp in enumerate(all_kp):
        plt.plot(sfrs,kp,linestyle=linestyles[i],label=labels[i],marker='s')
    plt.gca().set_xticks(np.linspace(0,2,11))
    plt.xlim(0,2)
    plt.ylim(0,1)
    plt.xlabel('$n_{\\rm SFR}$')
    plt.ylabel('$p_{\\rm KS}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_vbar_dist(sfrs,all_vbars,labels,outfile):
    """
    Plots the distribution of ks p-valuesas a function of alphs
    """
    plt.figure()
    linestyles=["-","--","-.",":"]
    for i,vbar in enumerate(all_vbars):
        print("Vbar at z=0",vbar[0],labels[i])
        plt.plot(sfrs,vbar,linestyle=linestyles[i],label=labels[i],marker='s')
    plt.gca().set_xticks(np.linspace(0,2,11))
    plt.xlim(0,2)
    plt.ylim(0.4,0.7)
    plt.xlabel('$n_{\\rm SFR}$')
    plt.ylabel('$\\left< V/V_{\\rm max} \\right>$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
def do_ks_stats(infile,histname,ksname):
    """
    Calculates a ks stat using an input file
    
    Infile is the file to read data from
    histname is the output file for the histogram
    ksname is the filename for the ks plot
    
    """
    VVmax = read_data(infile)
    VVmax = np.sort(VVmax)
    vbar = np.sum(VVmax)/VVmax.size
    
    # sets up the cumulative distribution for ks test
    NFRB = VVmax.size
    yvals = np.linspace(0.,1.,NFRB)
    
    # performs ks test
    k,p = kstest(VVmax,uniform)
    
    # makes plots
    make_ks_plot(VVmax,ksname,p)
    make_vvmax_hist(VVmax,histname,p)
    return p,vbar

def make_ks_plot(vvmax,outfile,p):
    """
    Makes a cumulative distribution plot of VVmax values
    """
    
    svvmax = np.sort(vvmax)
    NFRB = svvmax.size
    xvals = np.zeros([2*NFRB+2])
    yvals = np.zeros([2*NFRB+2])
    
    xvals[0]=0.
    xvals[-1] = 1.
    yvals[-2] = 1.
    yvals[-1] = 1.
    for i in np.arange(NFRB):
        xvals[2*i+1] = svvmax[i]
        xvals[2*i+2] = svvmax[i]
        yvals[2*i] = i/NFRB
        yvals[2*i+1] = i/NFRB
    
    plt.figure()
    plt.plot(xvals,yvals,label='Observed: p='+str(p)[0:5])
    plt.plot([0.,1.],[0.,1.],color='black',linestyle='--',label='Uniform')
    plt.xlabel('$V/V_{\\rm max}$')
    plt.ylabel('cdf')
    plt.legend()
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    
def make_vvmax_hist(vvmax,outfile,p):
    """
    Plots a histogram of VVmax values
    """    
    bins = np.linspace(0,1,11)
    
    h,b = np.histogram(vvmax,bins)
    themax = int(np.max(h))
    
    
    
    plt.figure()
    plt.hist(vvmax,bins,label="Observed: p="+str(p)[0:5])
    plt.xlabel('$V/V_{\\rm max}$')
    plt.ylabel('$N_{\\rm FRB}$')
    plt.gca().set_yticks(np.arange(themax+2))
    plt.xlim(0,1)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()
    # plots a histogram

def uniform(x):
    """
    cdf for V/Vmax. It's just a uniform distribution
    """
    return x

def read_data(infile):
    """
    Reads in output files generated by calc_vvmax.py
    Returns only the V / Vmax values - all that's relevant
    
    Since v2, there is a chance that we have vmax == v. This
    is because I forgot to cut FRBs with z>zmax from the sample
    when doing the localisation. These hence have z_frb > zmax
    for all beam values. Thus, zmax > zDM[i] will never trigger,
    and zmax is identical for V and Vmax
    """
    data = np.loadtxt(infile)
    FRBs = data[:,0]
    JHz = data[:,1]
    zmaxB = data[:,2]
    zmaxC = data[:,3]
    V = data[:,4]
    Vmax = data[:,5]
    VVmax = data[:,6]
    
    for FRB in [20170428,20210407,20210912,20220610]:
        OK = np.where(FRBs != FRB)
        FRBs = FRBs[OK]
        JHz = JHz[OK]
        zmaxB = zmaxB[OK]
        zmaxC = zmaxC[OK]
        V = V[OK]
        Vmax = Vmax[OK]
        VVmax = VVmax[OK]
    return VVmax

main(load=False)
