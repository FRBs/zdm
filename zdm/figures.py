import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cmasher as cmr

from zdm import cosmology as cos

def plot_grid(
    zDMgrid,
    zvals,
    dmvals,
    zmax=1,
    DMmax=1000,
    norm=0,
    log=True,
    name="temp.pdf",
    label='$\\log_{10}p(DM_{\\rm EG},z)$',
    ylabel="${\\rm DM}_{\\rm EG}$ (pc cm$^{-3}$)",
    project=False,
    conts=False,
    FRBZs=None,
    FRBDMs=None,
    plt_dicts=None,
    cont_dicts=None,
    cmap=None,
    Aconts=None,
    Macquart=None,
    title=None,
    H0=None,
    showplot=False,
    DMlines=None,
    DMlims=None,
    clim=False,
    special=None,
    pdmgz=None,
    save=True,
    othergrids=None,
    othernames=None
):
    """
    Very complicated routine for plotting 2D zdm grids 

    Args:
        zDMgrid (2D array): P(z,DM) grid
        zvals (1D array): z values corresponding to zDMgrid
        dmvals (1D array): DM values corresponding to zDMgrid
        zmax (int, optional): Maximum z value to display
        DMmax (int, optional): Maximum DM value to display
        norm (int, optional): Method to normalise zDMgrid.
                                0: No normalisation
                                1: Normalise by dm bin
                                2: Normalise by sum of zDMgrid
                                3: Normalise by max value of zDMgrid
                                4: Set peak value at each z to unity
        log (bool, optional): Plot P(z,DM) in log space
        name (str, optional): Outfile name
        label (str, optional): Colourbar label
        ylabel (str,optional): Label on y axis of plot
        project (bool, optional): Add projections of P(z) and P(DM)
        conts (bool, optional): create contours in probability p(dm|z),
            at fractional levels set by conts. Defaults to False.
        FRBZs (list of 1D arrays, optional): List of FRB Zs to plot
            (each list can have customised plotting styles, e.g. for different surveys)
        FRBDMs (list of 1D arrays, optional): List of FRB DMs to plot (corrseponding to FRBZs)
        plt_dicts (list of dictionaries, optional): List of dictionaries
                containing the plotting parameters for each 'set' of data points
                (corresponding to FRBZs and FRBDMs). E.g. can contain marker, color, label etc
        cmap (str, optional): Alternate color map for PDF
        Aconts (bool, optional): Create contours in 2D probabilty space, at fractional
                    levels set by Aconts. Defaults to False.
        Macquart (state, optional): state object.  Used to generate the Maquart relation.
            Defaults to None, i.e. do not show the Macquart relation.
        title (str, optional): Title of the plot
        H0 ([type], optional): [description]. Defaults to None.
        showplot (bool, optional): use plt.show to show plot. Defaults to False.
        DMlines (list, optional): plot lines for unlocalised FRBs at these DMs
        DMlims (list, optional): plot horizontal lines to indicate the
                        maximum searched DM of a given survey
        clim ([float,float], optional): pair of floats giving colorbar limits.
            Defaults to False (automatic limit)
        special(list,optional): list of [z,dm] values to show as a special big star
        pdmgz(list of floats, optional): a list of cumulative values of p(DM|z) to
            plot. Must range from 0 to 1.
        othergrids (list of grids) [None]: a list of grids to plot contours for. Uses
            Aconts
        othernames (list of names) [None]: list of names for original *and* other grid.
            Used only if othergrids is not None. Must be length of othergrids +1.
    """
    if H0 is None:
        H0 = cos.cosmo.H0
    if cmap is None:
        # cmx = plt.get_cmap("cubehelix")
        cmap = cmr.prinsenvlag_r
    else:
        cmap = plt.get_cmap(cmap)

    # Set default colors
    if plt_dicts == None and FRBDMs is not None:
        p_cmap = cmr.arctic
        data_clrs = p_cmap(np.linspace(0.2, 0.8, len(FRBDMs)))
        plt_dicts = [{'color': clr, 'marker': 'o'} for clr in data_clrs]
    elif isinstance(plt_dicts, dict):
        plt_dicts = [plt_dicts]

    if Aconts:
        linestyles = ['--', '-.', ':', '-']
        c_cmap = cmr.arctic
        if othergrids is not None:
            n_conts = len(Aconts) + len(othergrids)
        else:
            n_conts = len(Aconts)
        cont_clrs = c_cmap(np.linspace(0.2, 0.8, n_conts))

        # Make dictionary for the contours
        if cont_dicts == None:
            cont_dicts = [{'colors': [cont_clrs[i]], 'linestyles': [linestyles[i % len(linestyles)]]} for i in range(len(cont_clrs))]
            
        # Make dictionary for the legend
        l_cont_dicts = [cont_dict.copy() for cont_dict in cont_dicts]
        for i in range(len(l_cont_dicts)):
            l_cont_dicts[i]['color'] = l_cont_dicts[i]['colors'][0]
            del l_cont_dicts[i]['colors']
            l_cont_dicts[i]['linestyle'] = l_cont_dicts[i]['linestyles'][0]
            del l_cont_dicts[i]['linestyles']

    ##### imshow of grid #######

    # we protect these variables
    zDMgrid = np.copy(zDMgrid)
    zvals = np.copy(zvals)
    dmvals = np.copy(dmvals)

    if project:
        fig = plt.figure(1, figsize=(8, 8))
        
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        gap = 0.02
        woff = width + gap + left
        hoff = height + gap + bottom
        dw = 1.0 - woff - gap
        dh = 1.0 - hoff - gap

        delta = 1 - height - bottom - 0.05
        gap = 0.11
        rect_2D = [left, bottom, width, height]
        rect_1Dx = [left, hoff, width, dh]
        rect_1Dy = [woff, bottom, dw, height]
        rect_cb = [woff, hoff, dw * 0.5, dh]
        ax1 = plt.axes(rect_2D)
        axx = plt.axes(rect_1Dx)
        axy = plt.axes(rect_1Dy)
        acb = plt.axes(rect_cb)
    else:
        fig,ax1 = plt.subplots()
    
    plt.sca(ax1)
    
    plt.xlabel("z")
    plt.ylabel(ylabel)
    
    nz, ndm = zDMgrid.shape
    
    # attenuate grids in x-direction
    ixmax = np.where(zvals > zmax)[0]
    if len(ixmax) > 0:
        zvals = zvals[: ixmax[0]]
        nz = zvals.size
        zDMgrid = zDMgrid[: ixmax[0], :]
        if othergrids:
            for grid in othergrids:
                grid = grid[: ixmax[0], :]
    
    # currently this is "per cell" - now to change to "per DM"
    # normalises the grid by the bin width, i.e. probability per bin, not probability density
    ddm = dmvals[1] - dmvals[0]
    dz = zvals[1] - zvals[0]
    if norm == 1:
        zDMgrid /= ddm
        if othergrids is not None:
            for grid in othergrids:
                grid /= ddm
        # if Aconts:
        #    alevels /= ddm
    elif norm == 2:
        xnorm = np.sum(zDMgrid)
        zDMgrid /= xnorm
        if othergrids is not None:
            for grid in othergrids:
                grid /= np.sum(grid)
        # if Aconts:
        #    alevels /= xnorm
    elif norm == 3:
        zDMgrid /= np.max(zDMgrid)
        if othergrids is not None:
            for grid in othergrids:
                grid /= np.max(grid)
    elif norm == 4:
        # normalise by peak value in p(DM|z))
        peaks = np.max(zDMgrid,axis=1)
        zDMgrid = (zDMgrid.T / peaks).T
        if othergrids is not None:
            for grid in othergrids:
                peaks = np.max(grid,axis=1)
                grid = (grid.T / peaks).T
        
    # sets up to plot contour-like things as a function of p(dm given z)
    if pdmgz is not None:
        # gets all values where zsum is not zero
        z1d = np.sum(zDMgrid,axis=1) # sums over DM
        OK = np.where(z1d > 0.)[0]
        pdmgz_z = zvals[OK]
        pdmgz_cs = np.cumsum(zDMgrid[OK,:],axis=1)
        pdmgz_dm = np.zeros([pdmgz_z.size, len(pdmgz)])
        for iz,z in enumerate(pdmgz_z):
            this_cs = pdmgz_cs[iz,:]/pdmgz_cs[iz,-1]
            for iv,val in enumerate(pdmgz):
                i1 = np.where(this_cs < val)[0][-1]
                i2 = i1+1
                k2 = (val - this_cs[i1])/(this_cs[i2] - this_cs[i1])
                k1 = 1.-k2
                dmval = k1*dmvals[i1] + k2*dmvals[i2]
                pdmgz_dm[iz,iv] = dmval
    
    # sets contours according to norm
    if Aconts:
        
        alevels = get_alevels(zDMgrid,Aconts)
        if norm == 1:
            alevels /= ddm
        elif norm == 2:
            alevels /= xnorm
        
        if othergrids is not None:
            other_alevels=[]
            for grid in othergrids:
                other_alevels.append(get_alevels(grid,Aconts))

    ### generates contours *before* cutting array in DM ###
    ### might need to normalise contours by integer lengths, oh well! ###
    if conts:
        nc = len(conts)
        carray = np.zeros([nc, nz])
        for i in np.arange(nz):
            cdf = np.cumsum(zDMgrid[i, :])
            cdf /= cdf[-1]

            for j, c in enumerate(conts):
                less = np.where(cdf < c)[0]

                if len(less) == 0:
                    carray[j, i] = 0.0
                    dmc = 0.0
                    il1 = 0
                    il2 = 0
                else:
                    il1 = less[-1]

                    if il1 == ndm - 1:
                        il1 = ndm - 2

                    il2 = il1 + 1
                    k1 = (cdf[il2] - c) / (cdf[il2] - cdf[il1])
                    dmc = k1 * dmvals[il1] + (1.0 - k1) * dmvals[il2]
                    carray[j, i] = dmc

        ddm = dmvals[1] - dmvals[0]
        carray /= ddm  # turns this into integer units for plotting

    iymax = np.where(dmvals > DMmax)[0]
    if len(iymax) > 0:
        dmvals = dmvals[: iymax[0]]
        zDMgrid = zDMgrid[:, : iymax[0]]
        ndm = dmvals.size
        if othergrids:
            for i,grid in enumerate(othergrids):
                othergrids[i] = grid[:, : iymax[0]]
    
    # now sets the limits to the actual size of the grid
    NX,NY = zDMgrid.shape
    plt.xlim(0,NX)
    plt.ylim(0,NY)
    
    if log:
        # checks against zeros for a log-plot
        orig = np.copy(zDMgrid)
        zDMgrid = zDMgrid.reshape(zDMgrid.size)
        setzero = np.where(zDMgrid == 0.0)
        zDMgrid = np.log10(zDMgrid)
        zDMgrid[setzero] = -100
        zDMgrid = zDMgrid.reshape(nz, ndm)
        if Aconts:
            alevels = np.log10(alevels)
    else:
        orig = zDMgrid

    # gets a square plot
    aspect = nz / float(ndm)

    # sets the x and y tics. These are now bin edges
    
    xtvals = np.arange(zvals.size+1)
    xtlabels = np.linspace(0.,zvals[0]+zvals[-1],zvals.size+1)
    everx = int(zvals.size / 5)
    # adds xticks at "edges"
    xtvals[-1] *= 0.999 # just allows it to squeeze on
    plt.xticks(xtvals[0 :: everx], xtlabels[0 :: everx])
    
    ytvals = np.arange(dmvals.size+1)
    ytvals[-1] *= 0.999 # just allows it to squeeze on
    ytlabels = np.linspace(0.,dmvals[0]+dmvals[-1],dmvals.size+1)
    every = int(dmvals.size / 5)
    plt.yticks(ytvals[0 :: every], ytlabels[0 :: every])

    im = plt.imshow(
        zDMgrid.T, cmap=cmap, origin="lower", interpolation="None", aspect=aspect
    )
    
    # plots "A"contours (i.e., over "Amplitudes"). Doing so for multiple grids
    # if necessary
    # NOTE: currently no way to plot contour labels, hence the use of dummy plots
    if Aconts:
        ax = plt.gca()
        cs = ax.contour(
            zDMgrid.T, levels=alevels, origin="lower", linewidths=2, linestyles=linestyles, colors=cont_clrs
            # zDMgrid.T, levels=alevels, **cont_dicts
        )
        cntrs=[cs]
        if othernames is not None:
            h,=plt.plot([-1e6,-2e6],[-1e6,-2e6],**l_cont_dicts[0],label=othernames[0])
            handles=[h]
        else:
            handles=[]
            for iA,Alevel in enumerate(Aconts):
                    h,=plt.plot([-1e6,-2e6],[-1e6,-2e6],**l_cont_dicts[iA],label=str(1.-Alevel)+"%")
            handles.append(h)
        
        if othergrids is not None:
            for i,grid in enumerate(othergrids):
                print("size of i in othergrids is ",i)
                cntr = ax.contour(grid.T, levels=other_alevels[i], origin="lower",
                    **cont_dicts[i+1])
                if othernames is not None:
                    #make a dummy plot
                    h,=plt.plot([-1e6,-2e6],[-1e6,-2e6], **l_cont_dicts[i+1],label=othernames[i+1])
                    #h,=plt.plot([-1e6,-2e6],[-1e6,-2e6],linestyle=styles[i+1], marker=plt_dicts[i+1]['marker'], 
                    #    markeredgewidth=plt_dicts[i+1]['markeredgewidth'], color=cont_colours[i+1],label=othernames[i+1])
                    handles.append(h)
    
            plt.legend(handles=handles,loc="lower right")
    
    
    ###### gets decent axis labels, down to 1 decimal place #######
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    for i in np.arange(len(labels)):
        labels[i] = labels[i][0:4]
    ax.set_xticklabels(labels)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    for i in np.arange(len(labels)):
        if "." in labels[i]:
            labels[i] = labels[i].split(".")[0]
    ax.set_yticklabels(labels)
    ax.yaxis.labelpad = 0

    # draw horizontal lines for a fixed DM
    if DMlines is not None:
        if log:
            tempgrid = np.copy(zDMgrid)
            tempgrid = zDMgrid - np.max(zDMgrid)
            tempgrid = 10.0 ** zDMgrid
        else:
            tempgrid = zDMgrid
        for DM in DMlines:
            if DM > np.max(dmvals):
                print(
                    "Cannot draw DM line ",
                    DM,
                    " - range ",
                    np.max(dmvals),
                    " too small...",
                )
                continue
            # determines how far to draw line
            iDM2 = np.where(dmvals > DM)[0][0]  # lowest value
            iDM1 = iDM2 - 1
            kDM = (DM - dmvals[iDM1]) / (dmvals[iDM2] - dmvals[iDM1])
            cDM1 = np.cumsum(tempgrid[:, iDM1])
            cDM1 /= cDM1[-1]
            cDM2 = np.cumsum(tempgrid[:, iDM2])
            cDM2 /= cDM2[-1]
            
            stop1 = np.where(cDM1 < 0.99)[0][-1]
            stop2 = np.where(cDM2 < 0.99)[0][-1]
            zstop = kDM * zvals[stop2] + (1.0 - kDM) * zvals[stop1]
            zstop /= zvals[1] - zvals[0]
            DM /= dmvals[1] - dmvals[0]
            plt.plot([0, zstop], [DM, DM], color=plt_dicts[0]['color'], linestyle=":")

    if DMlims is not None:
        for DMlim in DMlims:
            if DMlim is not None and DMlim < DMmax:
                DMlim /= dmvals[1] - dmvals[0]
                ax.axhline(DMlim, 0, 1, color='k', linestyle="-")

    # performs plots for the pdmgz variable
    if pdmgz is not None:
        styles = ['-','-','-']
        widths = [2,3,2]
        plt.ylim(0,ndm-1)
        plt.xlim(0,nz-1)
        # now converts to plot units [urgh...]
        plot_z = np.arange(pdmgz_z.size)
        for iv,val in enumerate(pdmgz):
            plot_dm = pdmgz_dm[:,iv]/ddm # plot is in integer units
            plt.plot(plot_z,plot_dm,linestyle=styles[iv],linewidth=widths[iv],color='white')
    
    # plots contours i there
    if conts:
        cont_styles=[":","-","--","-."]
        plt.ylim(0, ndm - 1)
        for i in np.arange(nc):
            cstyle = i%4
            j = int(nc - i - 1)
            plt.plot(np.arange(nz), carray[j, :], label=str(int(conts[j]*100))+"%", color="white",\
                    linestyle=cont_styles[cstyle])
        l = plt.legend(loc="upper left", fontsize=8)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        for text in l.get_texts():
            text.set_color("white")

    if Macquart is not None:
        # Note this is the Median for the lognormal, not the mean
        muDMhost = np.log(10 ** Macquart.host.lmean)
        sigmaDMhost = np.log(10 ** Macquart.host.lsigma)
        meanHost = np.exp(muDMhost + sigmaDMhost ** 2 / 2.0)
        medianHost = np.exp(muDMhost)
        # print(f"Host: mean={meanHost}, median={medianHost}")
        plt.ylim(0, ndm - 1)
        plt.xlim(0, nz - 1)
        zmax = zvals[-1]
        nz = zvals.size
        # DMbar, zeval = igm.average_DM(zmax, cumul=True, neval=nz+1)
        DM_cosmic = pcosmic.get_mean_DM(zvals, Macquart)

        # idea is that 1 point is 1, hence...
        zeval = zvals / dz
        DMEG_mean = (DM_cosmic + meanHost) / ddm
        DMEG_median = (DM_cosmic + medianHost) / ddm
        plt.plot(
            zeval,
            DMEG_mean,
            color="blue",
            linewidth=2,
            label="Macquart relation (mean)",
        )
        # removed median, because it is only media of HOST not DM cosmic
        # plt.plot(zeval,DMEG_median,color='blue',
        #         linewidth=2, ls='--',
        #         label='Macquart relation (median)')
        l = plt.legend(loc="lower right", fontsize=12)
        # l=plt.legend(bbox_to_anchor=(0.2, 0.8),fontsize=8)
        # for text in l.get_texts():
        # 	text.set_color("white")

    # limit to a reasonable range if logscale
    
    if log:
        themax = np.nanmax(zDMgrid)
        themin = int(themax - 4)
        themax = int(themax)
        plt.clim(themin, themax)
    
    if clim:
        plt.clim(clim[0], clim[1])
    
    ##### add FRB host galaxies at some DM/redshift #####
    if FRBZs is not None and len(FRBZs) != 0:
        if hasattr(FRBZs[0], "__len__"):
            # we are dealing with a list of lists from multiple surveys
            for i, FRBZ in enumerate(FRBZs):
                # test if this is a list of FRBZs or a list of lists
                
                if FRBZ is not None and len(FRBZ) != 0:
                    FRBDM = FRBDMs[i]
                    iDMs = FRBDM / ddm
                    iZ = FRBZ / dz
                    OK = np.where(FRBZ > 0)[0]
                    plt.plot(iZ[OK], iDMs[OK], linestyle="", **plt_dicts[i])
        else:
            # just a single list of values
            OK = np.where(FRBDMs > 0)[0]
            iDMs = FRBDMs / ddm
            iZ = FRBZs / dz
            plt.plot(iZ[OK], iDMs[OK], 'ro',linestyle="")
            
    legend = plt.legend(loc='upper left')
    # legend = plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=12, markerscale=1, ncol=2)
    # legend.get_frame().set_facecolor('lightgrey')

    if special is not None:
        iDM = special[0] / ddm
        iz = special[1] / dz
        plt.plot([iz], [iDM], "*", markersize=10, color="blue", linestyle="")

    # do 1-D projected plots
    if project:
        plt.sca(acb)
        cbar = plt.colorbar(
            im, fraction=0.046, shrink=1.2, aspect=20, pad=0.00, cax=acb
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(label, fontsize=8)
        
        axy.set_yticklabels([])
        axy.set_ylim(0,DMmax)
        axx.set_xlim(0,zmax)
        # axy.set_xticklabels([])
        # axx.set_yticklabels([])
        axx.set_xticklabels([])
        yonly = np.sum(orig, axis=0)
        xonly = np.sum(orig, axis=1)

        axy.plot(yonly, dmvals)  # DM is the vertical axis now
        axx.plot(zvals, xonly)

        # if plotting DM only, put this on the axy axis showing DM distribution
        if FRBDMs is not None:
            if hasattr(FRBDMs[0], "__len__"):
                # dealing with a list of lists
                for FRBDM in FRBDMs:
                    if FRBDM is not None:
                        hvals=np.zeros(FRBDM.size)
                        for i,DM in enumerate(FRBDM):
                            if DM > dmvals[-1]:
                                havls[i] = 0
                            else:
                                hvals[i] = yonly[np.where(dmvals > DM)[0][0]]
                    
                        axy.plot(hvals,FRBDM,'ro', linestyle="")
                        for tick in axy.yaxis.get_major_ticks():
                            tick.label1.set_fontsize(6)
            else:
                hvals=np.zeros(FRBDMs.size)
                for i,DM in enumerate(FRBDMs):
                    if DM > dmvals[-1]:
                        havls[i] = 0
                    else:
                        hvals[i] = yonly[np.where(dmvals > DM)[0][0]]
                axy.plot(hvals,FRBDMs,'ro',linestyle="")
                
        if FRBZs is not None:
            if hasattr(FRBZs[0], "__len__"):
                # dealing with a list of lists
                for FRBZ in FRBZs:
                    if FRBZ is not None:
                        OK = np.where(FRBZ > 0)[0]
                        hvals = np.zeros(FRBZ[OK].size)
                        for i, Z in enumerate(FRBZ[OK]):
                            hvals[i] = xonly[np.where(zvals > Z)[0][0]]
                        axx.plot(FRBZ[OK], hvals, "ro", linestyle="")
                        for tick in axx.xaxis.get_major_ticks():
                            tick.label1.set_fontsize(6)
            else:
                OK = np.where(FRBZs > 0)[0]
                hvals = np.zeros(FRBZs[OK].size)
                for i, Z in enumerate(FRBZs[OK]):
                    hvals[i] = xonly[np.where(zvals > Z)[0][0]]
                axx.plot(FRBZs[OK], hvals, "ro", linestyle="")
    else:
        cbar = plt.colorbar(im, fraction=0.046, shrink=1.2, aspect=15, pad=0.05)
        cbar.set_label(label)
        plt.tight_layout()

    if title is not None:
        plt.title(title)
    
    # checks if we still need the legend
    h,l = ax.get_legend_handles_labels()
    if len(h) == 0:
        # no handles in legend
        ax.get_legend().remove()
    if save:
        plt.tight_layout()
        plt.savefig(name, dpi=300, bbox_inches='tight')
    if showplot:
        plt.show()
    
    plt.close()


def get_alevels(zDMgrid,Aconts):
    """
    Gets contour levels giving 
    
    Grid: inoput zDM grid
    Aconts: list of contour levels giving %
    
    """
    slist = np.sort(zDMgrid.flatten())
    cslist = np.cumsum(slist)
    cslist /= cslist[-1]
    nAc = len(Aconts)
    alevels = np.zeros([nAc])
    for i, ac in enumerate(Aconts):
        # cslist is the cumulative probability distribution
        # Where cslist > ac determines the integer locations
        #    of all cells exceeding the threshold
        # The first in this list is the first place exceeding
        #    the threshold
        # The value of slist at that point is the
        #    level of the countour to draw
        iwhich = np.where(cslist > ac)[0][0]
        alevels[i] = slist[iwhich]
    return alevels


def find_Alevels(pgrid:np.ndarray,
                 Aconts:list, 
                 norm:bool=True,
                 log:bool=True):
    slist=np.sort(pgrid.flatten())
    cslist=np.cumsum(slist)
    cslist /= cslist[-1]
    nAc=len(Aconts)
    alevels=np.zeros([nAc])
    for i,ac in enumerate(Aconts):
        # cslist is the cumulative probability distribution
        # Where cslist > ac determines the integer locations
        #    of all cells exceeding the threshold
        # The first in this list is the first place exceeding
        #    the threshold
        # The value of slist at that point is the
        #    level of the countour to draw
        iwhich=np.where(cslist > ac)[0][0]
        alevels[i]=slist[iwhich]

    # Normalize?
    if norm:
        xnorm=np.sum(pgrid)
        alevels /= xnorm

    # Log?
    if log:
        alevels = np.log10(alevels)

    # Return
    return alevels


def proc_pgrid(pgrid:np.ndarray, 
               ivals:np.ndarray, imnx:tuple, 
               jvals:np.ndarray, jmnx:tuple, 
               norm:bool=True, log:bool=True):

    # Work on a copy
    proc_grid = pgrid.copy()

    # Norm first
    if norm:
        xnorm=np.sum(proc_grid)
        proc_grid /= xnorm

    # Cuts
    i_idx = (ivals > imnx[0]) & (ivals <= imnx[1])
    j_idx = (jvals > jmnx[0]) & (jvals <= jmnx[1])

    cut_ivals = ivals[i_idx]
    cut_jvals = jvals[j_idx]

    proc_grid = proc_grid[i_idx,:]
    proc_grid = proc_grid[:, j_idx]

    # Log?
    if log:
        neg = proc_grid <= 0.
        proc_grid = np.log10(proc_grid)
        proc_grid[neg] = -100.

    # Return
    return cut_ivals, cut_jvals, proc_grid

def ticks_pgrid(vals, everyn=5, fmt=None, these_vals=None):
    """ Generate ticks for one of the P(x,x,x) grids

    Args:
        vals (_type_): _description_
        everyn (int, optional): _description_. Defaults to 5.
        fmt (_type_, optional): _description_. Defaults to None.
        these_vals (list or np.ndarray, optional): Values to place
            the ticks at

    Returns:
        np.ndarray, np.ndarray:  Tick locations, values
    """
    if these_vals is None:
        tvals=np.arange(vals.size)
        everx=int(vals.size/everyn)
        tvals = tvals[everx-1::everx]
        ticks = vals[everx-1::everx]
    else:
        ticks = these_vals
        tvals = []
        for val in ticks:
            idx = np.argmin(np.abs(val-vals))
            tvals.append(idx)

    if fmt is None:
        pass
    elif fmt[0:3] == 'str':
        ticks = [str(item)[0:int(fmt[3:])] for item in ticks]
    elif fmt == 'int':
        ticks = [int(item) for item in ticks]
    # Return
    return tvals, ticks

