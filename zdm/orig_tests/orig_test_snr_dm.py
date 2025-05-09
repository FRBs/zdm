
"""
This tests various SNR/DM routines

Apologies, this is dodgy quick code - probably there are *way* too many import calls
"""

from zdm.MC_sample import loading
import os

import numpy as np

from zdm import iteration as it
from zdm import errors_misc_functions as err

import matplotlib.pyplot as plt
import matplotlib
from zdm import io

# setting some plotting defaults
matplotlib.rcParams['image.interpolation'] = None

defaultsize=14
ds=4
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : defaultsize}
matplotlib.rc('font', **font)

def main(N=100,plots=False):
    "created N*survey.NFRBs mock FRBs for all surveys"
    "Input N : creates N*survey.NFRBs"
    "Output : Sample(list) and Surveys(list)"
    
    ############## Load up ##############
    input_dict=io.process_jfile('../../papers/H0_I/Analysis/Cubes/craco_H0_Emax_cube.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)
    
    ######## choose which surveys to do an MC for #######
    
    # choose which survey to create an MC for
    names = ['CRAFT/FE', 'CRAFT/ICS', 'CRAFT/ICS892', 'PKS/Mb','CRAFT_CRACO_MC_alpha1_gamma_1000']
    dirnames=['ASKAP_FE','ASKAP_ICS','ASKAP_ICS892','Parkes_Mb','CRACO']
    # select which to perform an MC for
    
    Location="DM_SNR/"
    if not os.path.isdir(Location):
        os.mkdir(Location)
 
    ############## Initialise survey and grid ##############
    
    NSNR=11
    snrs=np.logspace(0,2,NSNR)
    
    if not os.path.isfile(Location+"psnrs.npy"):
        print("Could not find existing p(SNR,DM) data, generating...")
        
        s,g = loading.survey_and_grid(
            state_dict=state_dict,
            survey_name=names[4], NFRB=1000)
        
        dms=g.dmvals
        
        psnrs,dmpsnrs=err.get_sc_grid(g,snrs)
        
        np.save(Location+"psnrs.npy",psnrs)
        np.save(Location+"dmpsnrs.npy",dmpsnrs)
        np.save(Location+"dms.npy",dms)
        print("Generated data")
    else:
        print("Found existing p(SNR,DM) data, loading...")
        psnrs=np.load(Location+"psnrs.npy")
        dmpsnrs=np.load(Location+"dmpsnrs.npy")
        dms=np.load(Location+"dms.npy")
        print("loaded psnr data")
    print(dmpsnrs.shape)
    
    
    snrs=snrs[:-1]
    
    ### 1D figure ###
    plt.figure()
    plt.plot(snrs,psnrs*(snrs**1.5))
    plt.xlabel('s')
    plt.ylabel('$s^{2.5} p(s)$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-3,2e-3)
    plt.tight_layout()
    plt.savefig(Location+"psnr.pdf")
    plt.close()
    
    
    ### 2D figure - x-axis is DM ###
    
    plt.figure()
    plt.xlim(0,2000)
    for i,s in enumerate(snrs):
        norm=np.max(dmpsnrs[i,:])
        plt.plot(dms,dmpsnrs[i,:]/norm,label=str(s)[0:5])
    plt.xlabel('DM [pc/cm3]')
    plt.ylabel('$p({\\rm DM}|s)/p_{\\rm max}({\\rm DM}|s)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Location+"dm_by_snr.pdf")
    plt.close()
    
    plt.figure()
    plt.xlim(0,2000)
    for i,s in enumerate(snrs):
        plt.plot(dms,dmpsnrs[i,:],label=str(s)[0:5])
    plt.xlabel('DM [pc/cm3]')
    plt.ylabel('$p({\\rm DM}|s) [a.u.]$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Location+"nonorm_dm_by_snr.pdf")
    plt.close()
    
    ### 2D figure - x-axis is SNR ###
    
    # sums in DM intervals
    npieces=10
    ddm=dms[1]-dms[0]
    istep=int(200./ddm)
    Ddm=istep*ddm
    new_array=np.zeros([snrs.size,npieces])
    for i in np.arange(npieces):
        new_array[:,i] = np.sum(dmpsnrs[:,i*istep:(i+1)*istep],axis=1)
    
    
    plt.figure()
    for i in np.arange(npieces):
        label="DM: "+str(i*Ddm)+"-"+str((i+1)*Ddm)+"  [pc/cm3]"
        plt.plot(snrs,new_array[:,i]*snrs**1.5,label=label)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('s')
    plt.ylabel('$s^{2.5} p(s|{\\rm DM})$ [a.u.]')
    plt.legend(fontsize=10,loc="lower right")
    plt.tight_layout()
    plt.savefig(Location+"snr_by_dm.pdf")
    plt.close()
        
main()
