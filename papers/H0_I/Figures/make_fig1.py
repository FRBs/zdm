"""
Makes Figure 1 from the paper
Loads data from directory Figure1 and generates fig1.pdf there
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

def main():
    
    
    # loads data from CHIME catalogue 1
    file1 = open('Figure1/chimefrbcat1.csv', 'r')
    Lines = file1.readlines()
    scats=[]
    scaterrs=[]
    dms=[]
    widths=[]
    widtherrs=[]
    reps=[]
    chisqrs=[]
    for i,line in enumerate(Lines):
        words=line.split(',')
        if i==0:
            irep=words.index('repeater_name')
            idm=words.index('bonsai_dm')
            iscat=words.index('scat_time')
            iserr=words.index('scat_time_err')
            iwidth=words.index('width_fitb')
            ichi=words.index('chi_sq')
            continue
        
        reps.append(words[irep])
        if words[iscat][0]=='<':
            scats.append(float(words[iscat][1:]))
            scaterrs.append(-1)
        else:
            scats.append(float(words[iscat]))
            scaterrs.append(float(words[iserr]))
        
        if words[iwidth][0]=='<':
            widths.append(float(words[iwidth][1:]))
            widtherrs.append(-1)
        else:
            widths.append(float(words[iwidth]))
            widtherrs.append(1)
        
        chisqrs.append(float(words[ichi]))
        dms.append(float(words[idm]))
        
    
    scats=np.array(scats)
    scaterrs=np.array(scaterrs)
    widths=np.array(widths)
    widtherrs=np.array(widtherrs)
    chisqrs=np.array(chisqrs)
    dms=np.array(dms)
    
    once = [i for i, x in enumerate(reps) if x == "-9999"]
    
    scats *= 1000. #to ms
    widths *= 2000. # converts from sigma half-width to full width
    o_scats=scats[once]
    o_scaterrs=scaterrs[once]
    o_widths=widths[once]
    o_widtherrs=widtherrs[once]
    o_chisqrs=chisqrs[once]
    o_dms=dms[once]
    
    supper=np.where(o_scaterrs==-1)[0]
    
    sOK=np.where(o_scaterrs!=-1)[0]
    
    bins=np.linspace(0,20,101)
    plt.figure()
    plt.hist(o_scats[supper],bins=bins,label='CHIME upper limits',alpha=0.5)
    plt.hist(o_scats[sOK],bins=bins,label='CHIME scatter',alpha=0.5)
    plt.legend()
    plt.xlabel('$\\tau$ [600 MHz]')
    plt.ylabel('Occurences')
    plt.tight_layout()
    plt.savefig('Figure1/chime_scattering_times.pdf')
    plt.close()
    
    
    o_scats *= (0.6/1.3)**4.
    bins=np.linspace(0,10,101)
    
    
    ############ loads and manipulates data from Qiu etal 2020 #######
    hdata=np.loadtxt('Figure1/qiu_fits.dat')
    hwidths=hdata[:,1]
    hwidths *= 2. #Gaussian sigma to full width
    hdata=hdata[:,0]
    
    hupper=np.where(hdata<0.)
    hOK=np.where(hdata>0.)
    
    plt.figure()
    plt.hist(o_scats[supper],bins=bins,label='CHIME upper limits',alpha=0.3)
    plt.hist(o_scats[sOK],bins=bins,label='CHIME scatter',alpha=0.3)
    
    #plt.hist(-hdata[hupper],bins=bins,label='CRAFT upper limits',alpha=0.3)
    #plt.hist(hdata[hOK],bins=bins,label='CRAFT scatter',alpha=0.3)
    plt.xlim(0,3)
    #plt.ylim(0,20)
    plt.legend()
    plt.xlabel('$\\tau$ [1.3 GHz]')
    plt.ylabel('Occurences')
    plt.tight_layout()
    plt.savefig('Figure1/scaled_comparison_scattering_times.pdf')
    plt.close()
    
    plt.figure()
    #plt.hist(o_scats[supper],bins=bins,label='CHIME upper limits',alpha=0.3)
    #plt.hist(o_scats[sOK],bins=bins,label='CHIME scatter',alpha=0.3)
    bins=np.linspace(0,10,11)
    plt.hist(-hdata[hupper],bins=bins,label='CRAFT upper limits',alpha=0.3)
    plt.hist(hdata[hOK],bins=bins,label='CRAFT scatter',alpha=0.3)
    plt.xlim(0,10)
    plt.ylim(0,20)
    plt.legend()
    plt.xlabel('$\\tau$ [1.3 GHz]')
    plt.ylabel('Occurences')
    plt.tight_layout()
    plt.savefig('Figure1/craft_scattering_times.pdf')
    plt.close()
    
    hdata *= (1.3/0.9)**4
    o_scats *= (1.3/0.9)**4
    
    
    dms=np.loadtxt('Figure1/craft_892_frbs.dat')
    smear=8.3 * 1. * dms * (0.892)**-3 / 1000.
    d=False
    xmax=30
    bins=np.linspace(0,xmax,16)
    plt.figure()
    #plt.hist(o_scats[supper],bins=bins,label='CHIME upper limits',alpha=0.3,fill=False,linewidth=3,edgecolor='red',density=True)
    #plt.hist(o_scats[sOK],bins=bins,label='CHIME scatter',alpha=0.3,fill=False,linewidth=3,edgecolor='blue',density=True)
    
    plt.hist(-hdata[hupper],bins=bins,label='CRAFT upper limits',alpha=0.7,fill=False,linewidth=3,edgecolor='green',density=d,linestyle=":")
    plt.hist(hdata[hOK],bins=bins,label='CRAFT scatter',alpha=0.5,fill=False,linewidth=3,edgecolor='purple',density=d)
    
    plt.hist(smear,bins=bins,label='CRAFT 900MHz FRB DM smearing',alpha=1.0,fill=False,linewidth=1,edgecolor='black',density=d,linestyle='--')
    
    plt.xlim(0,xmax)
    #plt.ylim(0,20)
    plt.legend()
    plt.xlabel('$\\tau$ [1.3 GHz]')
    plt.ylabel('Occurences')
    plt.tight_layout()
    plt.savefig('Figure1/900_scaled_scattering_times.pdf')
    plt.close()
    
    
    ####### compares widths #######
    
    plt.figure()
    lim=np.where(hwidths<0.)[0]
    OK=np.where(hwidths>0.)[0]
    plt.hist(hwidths[OK], bins=np.linspace(0,5,11),label='ASKAP: Measured',alpha=0.5)
    plt.hist(hwidths[lim]*-1., bins=np.linspace(0,5,11),label='ASKAP: Limit',alpha=0.5)
    plt.hist(o_widths, bins=np.linspace(0,5,51),label='CHIME',alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Figure1/width_comparison.pdf')
    plt.close()

    ############################# CUMULATIVE DISTRIBUTIONS ######################
    #creates cumulative distribution for ASKAP
    #sets all limits to zero
    plt.figure()
    
    hmins=np.copy(hwidths)
    hmins[lim]=0.
    hminx,hminy=make_cumulative(hmins)
    
    hmaxs=np.copy(hwidths)
    hmaxs[lim] *= -1. #makes them 'normal'
    hmaxx,hmaxy=make_cumulative(hmaxs)
    
    # fill between does not work. Piece of shit
    #plt.fill_between(hminx,hminy,hmaxy,label='ASKAP',color='blue')
    plt.plot(hminx,hminy,label='ASKAP: measurements',linewidth=2)
    plt.plot(hmaxx,hmaxy,color=plt.gca().lines[-1].get_color(),linewidth=2)
    
    #hsorted=np.sort(hwidths[OK])
    #nlim=len(lim)
    #ntot=hsorted.size
    #vals=np.arange(ntot)+nlim+1.
    #vals = vals/(ntot+nlim+0.)
    
    #measured distribution: ASKAP
    hlogmean=0.98
    hlogsigma=0.73
    xbins=np.logspace(-3,3,601)
    logx=np.log(xbins)
    # equal log space
    yvals=np.exp(-0.5*(logx-hlogmean)**2./hlogsigma**2)
    yc=np.cumsum(yvals)
    yc /= yc[-1]
    plt.plot(xbins,yc,linestyle=':',linewidth=2,label='            fit, with Parkes')
    
    #true width distribution: ASKAP
    hlogmean=1.72
    hlogsigma=0.9
    xbins=np.logspace(-3,3,601)
    logx=np.log(xbins)
    # equal log space
    yvals=np.exp(-0.5*(logx-hlogmean)**2./hlogsigma**2)
    yc=np.cumsum(yvals)
    yc /= yc[-1]
    plt.plot(xbins,yc,linestyle='-.',linewidth=2,label='            bias corrected')
    
    
    ########same for CHIME ######
    
    # this one sets values at upper limits
    
    cmaxx,cmaxy=make_cumulative(o_widths)
    
    cminwidths=np.copy(o_widths)
    lims=np.where(o_widtherrs==-1)[0]
    cminwidths[lims]=0.
    cminx,cminy=make_cumulative(cminwidths)
    
    #csorted=np.sort(o_widths)
    #csorted *= 1000.
    #yc=np.linspace(0./csorted.size,1.,csorted.size)
    #plt.plot(csorted,yc,label='CHIME')
    plt.plot(cminx,cminy,label='CHIME: measurements',linewidth=2)
    plt.plot(cmaxx,cmaxy,color=plt.gca().lines[-1].get_color(),linewidth=2)
    
    
    
    plt.xlabel('Width [ms]')
    plt.ylabel('Cumulative probability')
    
    #true width distribution: CHIME
    clogmean=0.0
    clogsigma=0.97
    xbins=np.logspace(-3,3,601)
    logx=np.log(xbins)
    # equal log space
    yvals=np.exp(-0.5*(logx-clogmean)**2./clogsigma**2)
    yc=np.cumsum(yvals)
    yc /= yc[-1]
    # *2 factor accounts for the 'width' being a Gaussian half-width
    plt.plot(xbins*2,yc,linestyle='--',linewidth=2,label='            bias corrected')
    
    plt.xlim(0,10)
    plt.ylim(0,1)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('Figure1/figure1.pdf')
    plt.close()
  
def make_cumulative(values):
    """ makes a cumulative distribution plot of the values """
    
    nvals=values.size
    svalues=np.sort(values)
    x=np.zeros([nvals*2])
    y=np.zeros([nvals*2])
    
    y[::2]=np.arange(nvals)
    y[1::2]=(np.arange(nvals)+1)
    y /= nvals
    
    x[::2]=svalues
    x[1::2]=svalues
    return x,y
    
main()

