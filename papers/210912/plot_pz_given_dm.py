""" Calculate p(z|DM) for a given DM and survey
"""

# It should be possible to remove all the matplotlib calls from this
# but in the current implementation it is not removed.


def main(infile='pzgdm_210912.npz',outfile='210912_pz_given_dm.pdf'):
    
    import numpy as np
    from matplotlib import pyplot as plt
    
    # define plot
    plt.clf()
    ax = plt.gca()
    data=np.load(infile)
    
    zvals=data['zvals']
    all_pzgdm = data['all_pzgdm']
    nmodels,nz=all_pzgdm.shape
    
    for i in np.arange(nmodels):
        
        PzDM = all_pzgdm[i,:]
        
        if i==0:
            color='blue'
            lw=3
            style='-'
            label='Best fit'
        else:
            color='gray'
            lw=2
            style=':'
            label='90%'
        if i<2:
            ax.plot(zvals, PzDM, color=color,linestyle=style,\
                linewidth=lw,label=label)
        else:
            ax.plot(zvals, PzDM, color=color,linestyle=style,\
                linewidth=lw)
        
    
    # set ranges
    plt.ylim(0,np.max(all_pzgdm))
    
    # Limits
    #for z in [z_min, z_max]:
    #    ax.axvline(z, color='red', ls='--')

    ax.set_xlim(0, 2)

    ax.set_xlabel('z')
    ax.set_ylabel('P(z|DM) [Normalized]')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(outfile)
    plt.close()

main(infile='220610_pzgdm.npz',outfile='220610_pz_given_dm.pdf')
main(infile='210912_pzgdm.npz',outfile='210912_pz_given_dm.pdf')
