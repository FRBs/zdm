# standard python imports
import numpy as np
from astropy.cosmology import Planck18
from matplotlib import pyplot as plt
import os

# zdm imports
from zdm import loading
from zdm import parameters
from scipy.interpolate import RegularGridInterpolator
import pandas as pd

import h5py
import healpy as hp
from scipy.spatial import cKDTree
from ne2001 import density
from astropy.coordinates import SkyCoord
from astropy import units as u


from scipy.stats import gaussian_kde
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from zdm.zdm import figures





# path to galaxy catalog
gals_file = '/Users/lmasriba/Desktop/DMxFRB/MDPL2_UM_galaxy_lightcone_z_obs_0_1.52_dec_above_80.npz'

# path to CHIME survey grids
surdir = '/Users/lmasriba/FRBs/zdm/zdm/data/Surveys/CHIME/'

# LOS DMs from Konietzka et al. 2025
los_file = '/Users/lmasriba/Desktop/DMxFRB/Konietzka2025_DMmap_fullsky1_deep_v1.hdf5'


declinations = np.array([-10.6,-4.38, 6.083, 38.7, 78.67, 85.92, 90.0]) # declination bin edges for CHIME grids (degrees)


def get_grid():
    '''
    Main program to evaluate log0-likelihoods and predictions for
    repeat grids
    '''
    param_dict={'sfr_n': 0.21, 'alpha': 0.11, 'lmean': 2.18, 'lsigma': 0.42, 'lEmax': 41.37, 
                'lEmin': 39.47, 'gamma': -1.04, 'H0': 70.23, 'halo_method': 0, 'sigmaDMG': 0.0, 'sigmaHalo': 0.0,
                'lC': -7.61, 'min_lat': 0.0}

    opname="CHIME"
    opdir = opname+'/'
    
    if not os.path.exists(opdir):
        os.mkdir(opdir)
    
    # sets basic state and cosmology
    state = parameters.State()
    state.set_astropy_cosmo(Planck18)

    state.update_params(param_dict)
    
    
    # defines CHIME grids to load
    NDECBINS=6
    names=[]
    for i in np.arange(NDECBINS):
        name="CHIME_decbin_"+str(i)+"_of_6"
        names.append(name)
    survey_dir = surdir 
    ss,gs = loading.surveys_and_grids(survey_names=names, init_state=state, rand_DMG=False,sdir = survey_dir, repeaters=True)
    

    all_rates = []
    total  =gs[0].rates * 10**gs[0].state.FRBdemo.lC * ss[0].TOBS
    for i in range(NDECBINS):
        rate = gs[i].rates * 10**gs[i].state.FRBdemo.lC * ss[i].TOBS
        if i>0:
            total += rate
        all_rates.append(rate)

    # it is normalized by the maximum of the total rate -- same as in CHIME.py code 
    all_rates = np.array(all_rates)/np.max(total)


    figures.plot_grid(
                    total,
                    gs[0].zvals,
                    gs[0].dmvals,
                    norm=3,
                    #logrange=5,
                    log=True,
                    project=False,
                    showplot=True,
                    save=False,
                    zmax=2.5,
                    DMmax=2500,
                    Aconts=[0.01, 0.1, 0.5] #,
                    #cont_clrs=[1,1,1]
                    )


    # save normalized all_rates (plus grids) to compressed .npz for later use
    out_fname =  'CHIME_rates.npz'
    np.savez_compressed(out_fname,
                        all_rates=all_rates,
                        zvals=gs[0].zvals,
                        dmvals=gs[0].dmvals,
                        declinations=declinations)
    print(f"Saved all_rates to {out_fname}")
    lkadfsa

    
    # ---------------------------------------------------------------------
    


    def plot_log_rate_comparison():
        """
        Plots side-by-side log10 rate distributions for declination bins 0 and 5,
        with percentile contours overlaid.
        """
        # Select indices for dmvals 0-4000 and zvals 0-4
        dm_mask_0 = (gs[0].dmvals >= 0) & (gs[0].dmvals <= 4000)
        z_mask_0 = (gs[0].zvals >= 0) & (gs[0].zvals <= 4)
        dm_mask_2 = (gs[2].dmvals >= 0) & (gs[2].dmvals <= 4000)
        z_mask_2 = (gs[2].zvals >= 0) & (gs[2].zvals <= 4)

        # Subset the data
        rates0_sub = all_rates[0][np.ix_(z_mask_0, dm_mask_0)]
        rates2_sub = all_rates[2][np.ix_(z_mask_2, dm_mask_2)]
        zvals0_sub = gs[0].zvals[z_mask_0]
        dmvals0_sub = gs[0].dmvals[dm_mask_0]
        zvals2_sub = gs[2].zvals[z_mask_2]
        dmvals2_sub = gs[2].dmvals[dm_mask_2]

        # Compute log10 rates and find global vmin/vmax
        log_rates0 = np.log10(rates0_sub.T + 1e-10)
        log_rates2 = np.log10(rates2_sub.T + 1e-10)
        vmin = min(log_rates0.min(), log_rates2.min())
        vmax = max(log_rates0.max(), log_rates2.max())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for declination bin 0 (logarithmic rate)
        im0 = axes[0].imshow(
            log_rates0,
            origin='lower', aspect='auto',
            extent=[zvals0_sub[0], zvals0_sub[-1], dmvals0_sub[0], dmvals0_sub[-1]],
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[0].set_title('CHIME Declination Bin 0 (log10 Rate)')
        axes[0].set_xlabel('Redshift (z)')
        axes[0].set_ylabel('Dispersion Measure (DM)')
        fig.colorbar(im0, ax=axes[0], label='log10(Rate)')

        # Plot for declination bin 2 (logarithmic rate)
        im2 = axes[1].imshow(
            log_rates2,
            origin='lower', aspect='auto',
            extent=[zvals2_sub[0], zvals2_sub[-1], dmvals2_sub[0], dmvals2_sub[-1]],
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[1].set_title('CHIME Declination Bin 2 (log10 Rate)')
        axes[1].set_xlabel('Redshift (z)')
        axes[1].set_ylabel('Dispersion Measure (DM)')
        fig.colorbar(im2, ax=axes[1], label='log10(Rate)')

        # Overlay percentile contours on both subplots
        for ax, log_rates, zvals, dmvals in zip(
            axes, [log_rates0, log_rates2], [zvals0_sub, zvals2_sub], [dmvals0_sub, dmvals2_sub]
        ):
            thresholds = [
            np.percentile(10**log_rates, 90.),
            np.percentile(10**log_rates, 95.),
            np.percentile(10**log_rates, 99.)
            ]
            Z, DM = np.meshgrid(zvals, dmvals)
            contour = ax.contour(
            Z, DM, 10**log_rates, levels=thresholds,
            colors=['white', 'orange', 'red'], linewidths=2, linestyles=['dashed', 'dashdot', 'solid']
            )
            fmt = {thresholds[0]: '90%ile', thresholds[1]: '95%ile', thresholds[2]: '99%ile'}
            ax.clabel(contour, fmt=fmt, inline=True, fontsize=10)
        plt.tight_layout()
        plt.show()
        # Select indices for dmvals 0-4000 and zvals 0-4
        dm_mask_0 = (gs[0].dmvals >= 0) & (gs[0].dmvals <= 4000)
        z_mask_0 = (gs[0].zvals >= 0) & (gs[0].zvals <= 4)
        dm_mask_5 = (gs[5].dmvals >= 0) & (gs[5].dmvals <= 4000)
        z_mask_5 = (gs[5].zvals >= 0) & (gs[5].zvals <= 4)

        # Subset the data
        rates0_sub = all_rates[0][np.ix_(z_mask_0, dm_mask_0)]
        rates5_sub = all_rates[5][np.ix_(z_mask_5, dm_mask_5)]
        zvals0_sub = gs[0].zvals[z_mask_0]
        dmvals0_sub = gs[0].dmvals[dm_mask_0]
        zvals5_sub = gs[5].zvals[z_mask_5]
        dmvals5_sub = gs[5].dmvals[dm_mask_5]

        # Compute log10 rates and find global vmin/vmax
        log_rates0 = np.log10(rates0_sub.T + 1e-10)
        log_rates5 = np.log10(rates5_sub.T + 1e-10)
        vmin = min(log_rates0.min(), log_rates5.min())
        vmax = max(log_rates0.max(), log_rates5.max())

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot for declination bin 0 (logarithmic rate)
        im0 = axes[0].imshow(
            log_rates0,
            origin='lower', aspect='auto',
            extent=[zvals0_sub[0], zvals0_sub[-1], dmvals0_sub[0], dmvals0_sub[-1]],
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[0].set_title('CHIME Declination Bin 0 (log10 Rate)')
        axes[0].set_xlabel('Redshift (z)')
        axes[0].set_ylabel('Dispersion Measure (DM)')
        fig.colorbar(im0, ax=axes[0], label='log10(Rate)')

        # Plot for declination bin 5 (logarithmic rate)
        im5 = axes[1].imshow(
            log_rates5,
            origin='lower', aspect='auto',
            extent=[zvals5_sub[0], zvals5_sub[-1], dmvals5_sub[0], dmvals5_sub[-1]],
            cmap='viridis', vmin=vmin, vmax=vmax
        )
        axes[1].set_title('CHIME Declination Bin 5 (log10 Rate)')
        axes[1].set_xlabel('Redshift (z)')
        axes[1].set_ylabel('Dispersion Measure (DM)')
        fig.colorbar(im5, ax=axes[1], label='log10(Rate)')

        # Overlay percentile contours on both subplots
        for ax, log_rates, zvals, dmvals in zip(
            axes, [log_rates0, log_rates5], [zvals0_sub, zvals5_sub], [dmvals0_sub, dmvals5_sub]
        ):
            thresholds = [
                np.percentile(10**log_rates, 90.),
                np.percentile(10**log_rates, 95.),
                np.percentile(10**log_rates, 99.)
            ]
            Z, DM = np.meshgrid(zvals, dmvals)
            contour = ax.contour(
                Z, DM, 10**log_rates, levels=thresholds,
                colors=['white', 'orange', 'red'], linewidths=2, linestyles=['dashed', 'dashdot', 'solid']
            )
            fmt = {thresholds[0]: '90%ile', thresholds[1]: '95%ile', thresholds[2]: '99%ile'}
            ax.clabel(contour, fmt=fmt, inline=True, fontsize=10)
        plt.tight_layout()
        plt.show()

    # Call the plotting function
    #plot_log_rate_comparison()



    

    def plot_rate_contours():
        """
        Plots 2D contour comparison of rates for CHIME declination bins 0, 1, 3, 5.
        """
        # Calculate rates for gs[1], gs[3], and gs[5]
        rates_1 = gs[1].rates * 10**gs[1].state.FRBdemo.lC * ss[1].TOBS
        rates_3 = gs[3].rates * 10**gs[3].state.FRBdemo.lC * ss[3].TOBS
        rates_5 = gs[5].rates * 10**gs[5].state.FRBdemo.lC * ss[5].TOBS
        plt.figure(figsize=(10, 8))

        # Assuming rates arrays are 2D and have the same shape
        X, Y = np.meshgrid(gs[0].zvals, gs[0].dmvals)

        contour0 = plt.contour(X, Y, rates.T, levels=30, colors='blue', linestyles='solid')
        contour1 = plt.contour(X, Y, rates_1.T, levels=30, colors='red', linestyles='dashed')
        contour3 = plt.contour(X, Y, rates_3.T, levels=30, colors='green', linestyles='dashdot')
        contour5 = plt.contour(X, Y, rates_5.T, levels=30, colors='purple', linestyles='dotted')

        plt.clabel(contour0, inline=True, fontsize=8)
        plt.clabel(contour1, inline=True, fontsize=8)
        plt.clabel(contour3, inline=True, fontsize=8)
        plt.clabel(contour5, inline=True, fontsize=8)

        plt.title('Comparison of rates for CHIME declination bins 0, 1, 3, 5')
        plt.xlabel('Redshift (z)')
        plt.ylabel('Dispersion Measure (DM)')

        plt.tight_layout()
        plt.show()

    # Call the plotting function
    #plot_rate_contours(gs, rates, rates_1, rates_3, rates_5)

    # index range for z within [z_min, z_max] (uses global z_min, z_max)
    zvals = gs[0].zvals

    mask_max = zvals <= z_max
    zidx_max = int(np.where(mask_max)[0].max())

    mask_min = zvals >= z_min
    zidx_min = int(np.where(mask_min)[0].min())

    # print(f"z index range within [{z_min}, {z_max}] -> {zidx_min} (z={zvals[zidx_min]}) to {zidx_max} (z={zvals[zidx_max]})")

    # Trim all_rates to include only z bins between z_min and z_max (inclusive)
    all_rates = np.asarray(all_rates)  # ensure numpy array
    all_rates = all_rates[:, zidx_min:(zidx_max + 1), :]

    return all_rates, gs[0].zvals[zidx_min:(zidx_max + 1)], gs[0].dmvals

 # ---------------------------------------------------------------------


def interpolate_all_rates(declination_values, dm_values, z_values, all_rates):
        """
        Interpolates the all_rates array for given declination , dm, and z values.
        it return a probability of detecting an FRB at those values.

        Parameters:
            declination_values (array-like): declinations.
            dm_values (array-like): DM values to interpolate.
            z_values (array-like): z values to interpolate.
            all_rates (np.ndarray): Array of shape (NDECBINS, z, dm).
            gs (list): List of grid objects, one per declination bin.

        Returns:
            np.ndarray: probabilities of detecting an FRB.
        """
        results = []

        # check that values are within the declination range
        assert np.all(np.array(declination_values) < 90.0), "All declination_values must be less than 90.0"
        assert np.all(np.array(declination_values) > -10.6), "All declination_values must be greater than -10.6"

        # Define declination bin edges as an array of unique values
        declination_bin_edges = np.array([-10.6, -4.38, 6.083, 38.7, 78.67, 85.92, 90.0])
        declination_bins = np.digitize(declination_values, bins=declination_bin_edges) - 1

        z_grid = all_rates[0].shape[0]
        dm_grid = all_rates[0].shape[1]

        

        for dec_bin, dm, z in zip(declination_bins, dm_values, z_values):
            interpolator = RegularGridInterpolator(
                (z_grid, dm_grid), all_rates[dec_bin], bounds_error=False, fill_value=None
            )
            rate = interpolator([[z, dm]])[0]
            results.append(rate)
        return np.array(results)



# python
def get_gals(gal_file, test=False, max_rows=None):
    """
    Memory-efficient loader for the .npz galaxy catalog:
    - only reads necessary arrays
    - applies declination cut before constructing DataFrame
    - can downsample with max_rows to limit memory
    """
    npz = np.load(gal_file, mmap_mode='r')
    # read only required fields
    ral = npz['ral']
    decl = npz['decl']
    z_obs = npz['z_obs']
    obssm = npz['obssm'] if 'obssm' in npz.files else np.zeros_like(z_obs)
    obssfr = npz['obssfr'] if 'obssfr' in npz.files else np.zeros_like(z_obs)

    # apply declination cut before building DataFrame
    mask = decl > -10.6
    ral = ral[mask]
    decl = decl[mask]
    z_obs = z_obs[mask]
    obssm = obssm[mask]
    obssfr = obssfr[mask]

   

    N = len(z_obs)
    if test:
        # use tiny subset for testing
        sel = np.arange(min(1000, N))
    elif max_rows is not None and N > max_rows:
        # random downsample to max_rows
        idx = np.random.choice(N, size=max_rows, replace=False)
        sel = np.sort(idx)
    else:
        sel = np.arange(N)

    df = pd.DataFrame({
        'ral': ral[sel],
        'decl': decl[sel],
        'z_obs': z_obs[sel],
        'obssm': obssm[sel],
        'obssfr': obssfr[sel]
    })
    return df


def find_index(redshift, redshifts):
    """
    takes the target redshift and the redshifts array (e.g., z_continuous)
    returns index -- from Konietzka et al. 2025 notebook
    """
    index = np.where(np.round(redshifts,3)==np.round(redshift,3))[0][0]
    return index


def get_los(file,  test=False):
    ''''
    get LOS DMs from Konietzka et al. 2025 -- full sky
    '''
    with h5py.File(file, 'r') as f:
        dm = f['DMvalues'][:]
        z = f['redshifts'][:]

        DM_maps = np.array([])
        reds = np.array([])
        ras = np.array([])
        decs = np.array([])
 
        for i in range(0,len(z),50):
            z[i] = round(z[i],3)
            idx = find_index(z[i], z)
            DM_map = dm[idx, :]

            nside = hp.npix2nside(DM_map.size)
            pix_indices = np.arange(DM_map.size)
            theta, phi = hp.pix2ang(nside, pix_indices)

            # Convert theta, phi to declination and right ascension
            ra = np.degrees(phi)                   # Right Ascension in degrees
            dec = np.degrees(0.5 * np.pi - theta)  # Declination in degrees

            # Keep only sightlines with declination > -10.6
            mask = dec > -10.6
            ra = ra[mask]
            dec = dec[mask]
            DM_map = DM_map[mask]

            # Use the selected redshift value for all retained sightlines
            z_val = z[idx]
            z_arr = np.full(DM_map.size, z_val)

            DM_maps = np.concatenate([DM_maps, DM_map])
            reds =  np.concatenate([reds, z_arr])
            ras =  np.concatenate([ras, ra])
            decs =  np.concatenate([decs, dec])

        # Create a pandas DataFrame with columns: DM, z, ra, dec
        df = pd.DataFrame({
            'DM_cosmic': DM_maps,
            'z': reds,
            'ra': ras,
            'dec': decs
        })

    if test:
        df = df.iloc[:1000]  # use only first 1000 entries for testing

    return df
    

def match_gal_to_los(los_df, gal_df, zs):
    """
    For each galaxy in los_df, find the index of the closest entry in gal_df
    based on (ra, dec) position and zs.

    Returns:
        np.ndarray: Array of indices into gal_df for each galaxy in los_df.
    """

    # Match using RA, DEC and redshift with equal weight by standardizing each dimension.

    gal_z = gal_df['z_obs'].values

    gal_ra = gal_df['ral'].values
    gal_dec = gal_df['decl'].values

    los_ra = los_df['ra'].values
    los_dec = los_df['dec'].values
    los_z = np.asarray(zs)

    # compute combined means and stds so each dimension is given equal weight after standardization
    ra_all = np.concatenate([gal_ra, los_ra])
    dec_all = np.concatenate([gal_dec, los_dec])
    z_all = np.concatenate([gal_z, los_z])

    mean_ra, std_ra = ra_all.mean(), ra_all.std()
    mean_dec, std_dec = dec_all.mean(), dec_all.std()
    mean_z, std_z = z_all.mean(), z_all.std()

    # avoid zero / nan scales
    if std_ra == 0 or np.isnan(std_ra):
        std_ra = 1.0
    if std_dec == 0 or np.isnan(std_dec):
        std_dec = 1.0
    if std_z == 0 or np.isnan(std_z):
        std_z = 1.0

    # build standardized 3D points: [RA, DEC, z]
    gal_pts = np.vstack([
        (gal_ra - mean_ra) / std_ra,
        (gal_dec - mean_dec) / std_dec,
        (gal_z - mean_z) / std_z
    ]).T

    los_pts = np.vstack([
        (los_ra - mean_ra) / std_ra,
        (los_dec - mean_dec) / std_dec,
        (los_z - mean_z) / std_z
    ]).T

    tree = cKDTree(gal_pts)
    _, indices = tree.query(los_pts, k=1)
    return indices


def match_los_to_gal():
    '''
    find the closest los_df that matches gal_df ra, dec, z, and DM_cosmic to closest LOS in los_df
    Only search among LOS whose declination is within the same declination bin range
    (low_dec_edges[i], high_dec_edges[i]) for galaxy i. Falls back to a global tree
    if a bin contains no LOS.
    '''
    gal_ra = gal_df['ral'].values
    gal_dec = gal_df['decl'].values
    gal_z = gal_df['z_obs'].values
    dm = cosmic_dm

    los_ra = los_df['ra'].values
    los_dec = los_df['dec'].values
    los_z = los_df['z'].values
    los_dm = los_df['DM_cosmic'].values

    n = len(gal_ra)
    results = np.empty(n, dtype=int)

    # Build a global fallback KDTree (standardized by global scales)
    #scale_ra_all = los_ra.std() if (los_ra.std() != 0 and not np.isnan(los_ra.std())) else 1.0
    #scale_dec_all = los_dec.std() if (los_dec.std() != 0 and not np.isnan(los_dec.std())) else 1.0
    #scale_z_all = los_z.std() if (los_z.std() != 0 and not np.isnan(los_z.std())) else 1.0
    #scale_dm_all = los_dm.std() if (los_dm.std() != 0 and not np.isnan(los_dm.std())) else 1.0

    #pts_all = np.vstack([
    #    los_ra / scale_ra_all,
    #    los_dec / scale_dec_all,
    #    los_z / scale_z_all,
    #    los_dm / scale_dm_all
    #]).T
    #tree_all = cKDTree(pts_all)

    # Cache KDTree for each unique declination bin range
    tree_cache = {}
    for i in range(n):
        low = float(low_dec_edges[i])
        high = float(high_dec_edges[i])
        key = (low, high)

        if key in tree_cache:
            info = tree_cache[key]
        else:
            mask = (los_dec >= low) & (los_dec <= high)
            if not np.any(mask):
                # empty subset -> cache None to indicate fallback
                raise ValueError(f"No LOS entries found in declination range ({low}, {high})")
            else:
                sub_ra = los_ra[mask]
                sub_dec = los_dec[mask]
                sub_z = los_z[mask]
                sub_dm = los_dm[mask]

                s_ra = sub_ra.std() if (sub_ra.std() != 0 and not np.isnan(sub_ra.std())) else ((sub_ra.max() - sub_ra.min()) or 1.0)
                s_dec = sub_dec.std() if (sub_dec.std() != 0 and not np.isnan(sub_dec.std())) else ((sub_dec.max() - sub_dec.min()) or 1.0)
                s_z = sub_z.std() if (sub_z.std() != 0 and not np.isnan(sub_z.std())) else ((sub_z.max() - sub_z.min()) or 1.0)
                s_dm = sub_dm.std() if (sub_dm.std() != 0 and not np.isnan(sub_dm.std())) else ((sub_dm.max() - sub_dm.min()) or 1.0)

                pts = np.vstack([
                    sub_ra / s_ra,
                    sub_dec / s_dec,
                    sub_z / s_z,
                    sub_dm / s_dm
                ]).T
                tree = cKDTree(pts)
                idx_map = np.nonzero(mask)[0]  # map indices in subset -> indices in full los_df

                tree_cache[key] = (tree, s_ra, s_dec, s_z, s_dm, idx_map)
                info = tree_cache[key]

        # Query appropriate tree
        if info is None:
            raise ValueError(f"No LOS entries found in declination range ({low}, {high})")
        else:
            tree_subset, s_ra, s_dec, s_z, s_dm, idx_map = info
            q = np.array([gal_ra[i] / s_ra, gal_dec[i] / s_dec, gal_z[i] / s_z, dm[i] / s_dm])
            _, idx_sub = tree_subset.query(q, k=1)
            results[i] = int(idx_map[int(idx_sub)])

    return results.astype(int)
    





def get_lognorm_hosts(z_values):
    '''
    get host DMs from lognormal distribution as in Mas-Ribas & James 2025
    '''
    mu = 1.8
    sigma = 0.6
    # draw from lognormal distribution
    lognorm_dm = 10**np.random.normal(mu, sigma, size=len(z_values))
    # scale by (1+z)^-1 to the observer frame
    obs_lognorm_dm = lognorm_dm / (1 + z_values)
    return obs_lognorm_dm






def get_MW_DM(ra, dec):
    """
    Calculate the dispersion measure (DM) contribution from the Milky Way (MW) ISM
    along a given line of sight in galactic coordinates.
    Parameters:
        ra (array-like): Right Ascension values in degrees.
        dec (array-like): Declination values in degrees.
    Returns:
        float: The dispersion measure (DM) value calculated for the ISM along the 
        specified line of sight up to a distance of 100 parsecs.
    """

    coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

    gcoord = coord.transform_to('galactic')
    l, b = gcoord.l.value, gcoord.b.value

    
    ne = density.ElectronDensity()#**PARAMS)
    ismDM = [ne.DM(l[i], b[i], 1000.).value for i in range(len(l))]
    
    # Return
    return ismDM


def sampling(all_rates, num_samples):
    '''
    sample with rejection sampling from the input all_rates array
    '''
    # Flatten the all_rates array and create a mask for non-zero rates
    flattened_rates = all_rates.flatten()
    non_zero_ind = np.where(flattened_rates > 0)[0]

    probs = flattened_rates[non_zero_ind].astype(float) 
    probs /= np.sum(probs)  # Normalize to create a probability distribution

    # Sample from the distribution using rejection sampling
    ind_samples = np.random.choice(non_zero_ind, size=num_samples, p=probs)


    return ind_samples


def plot_zdm_sample(zs, dms):
    '''
    plot sampled z and DM values
    '''
    plt.figure(figsize=(8,6))
    plt.scatter(zs, dms, c='blue', alpha=0.7, edgecolors='k')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Dispersion Measure (DM)')
    plt.title('Sampled z and DM values from CHIME p(z, DM) grids')
    plt.xlim(0, z_max)
    plt.ylim(0, dm_samp.max()+100)
    plt.grid()
    plt.tight_layout()
    plt.show()


def find_los(zs, dms):
        """
        For each (z, dm) pair in zs, dms return the index of the closest row in the
        global los_df DataFrame based on (z, DM_cosmic) Euclidean distance,
        but only considering LOS entries whose declination lies within the
        corresponding low_dec_edges[i]..high_dec_edges[i] range.

        Requires global arrays low_dec_edges and high_dec_edges to be defined
        and of the same length as zs.
        """

        n_q = zs.size

        

        # Prepare global KDTree (fallback) with scaling
        los_z_all = los_df['z'].values
        los_dm_all = los_df['DM_cosmic'].values
        scale_z_all = np.std(los_z_all)
        scale_dm_all = np.std(los_dm_all)
        if scale_z_all == 0 or np.isnan(scale_z_all):
            scale_z_all = (los_z_all.max() - los_z_all.min()) or 1.0
        if scale_dm_all == 0 or np.isnan(scale_dm_all):
            scale_dm_all = (los_dm_all.max() - los_dm_all.min()) or 1.0
        los_points_all = np.vstack([los_z_all / scale_z_all, los_dm_all / scale_dm_all]).T
        tree_all = cKDTree(los_points_all)

        results = np.empty(n_q, dtype=int)

        # Cache KDTree for each unique declination bin range to avoid rebuilding repeatedly
        tree_cache = {}
        mask = None  # initialize mask variable
        for i in range(n_q):
            low_dec = low_dec_edges[i]
            high_dec = high_dec_edges[i]

            key = (float(low_dec), float(high_dec))
            if key in tree_cache:
                tree_info = tree_cache[key]
            else:
                mask = (los_df['dec'].values >= low_dec) & (los_df['dec'].values <= high_dec)
            if mask.sum() == 0:
                # empty subset: cache None to indicate fallback to global tree
                tree_cache[key] = None
                tree_info = None
            else:
                los_z = los_z_all[mask]
                los_dm = los_dm_all[mask]

                scale_z = np.std(los_z)
                scale_dm = np.std(los_dm)
                if scale_z == 0 or np.isnan(scale_z):
                    scale_z = (los_z.max() - los_z.min()) or 1.0
                if scale_dm == 0 or np.isnan(scale_dm):
                    scale_dm = (los_dm.max() - los_dm.min()) or 1.0

                pts = np.vstack([los_z / scale_z, los_dm / scale_dm]).T
                tree_subset = cKDTree(pts)
                idx_map = np.nonzero(mask)[0]  # map indices in subset -> indices in full los_df

                tree_cache[key] = (tree_subset, scale_z, scale_dm, idx_map)
                tree_info = tree_cache[key]

            # Query appropriate tree
            if tree_info is None:
                # fallback to global tree
                q = np.array([zs[i] / scale_z_all, dms[i] / scale_dm_all])
                _, idx_full = tree_all.query(q, k=1)
                results[i] = int(idx_full)
            else:
                tree_subset, scale_z, scale_dm, idx_map = tree_info
                q = np.array([zs[i] / scale_z, dms[i] / scale_dm])
                _, idx_sub = tree_subset.query(q, k=1)
                results[i] = int(idx_map[int(idx_sub)])

        return results


def plot_2dhists():
    '''
    plot 2D histograms of sampled z and DM vs mock sightlines
    '''
    
    # create a figure with two subplots: left = sampled grid, right = mock sightlines
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # left: 2D histogram of sampled z and DM
    xmin = 0.0
    xmax = max(z_samp.max(), z_mock.max()) + 0.1
    ymin = 0.0
    ymax = max( DM_extragalactic.max(), dm_samp.max()) + 100.0
    bins = [60, 60]

    H0, xedges0, yedges0, im0 = axes[0].hist2d(
        z_samp, dm_samp, bins=bins, range=[[xmin, xmax], [ymin, ymax]], cmap='viridis'
    )
    axes[0].set_xlabel('Redshift (z)')
    axes[0].set_ylabel('Dispersion Measure (DM)')
    axes[0].set_title('Sampled z and DM from CHIME p(z,DM) grids')
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[0].grid(True)

    H1, xedges1, yedges1, im1 = axes[1].hist2d(
        z_mock, DM_extragalactic, bins=bins, range=[[xmin, xmax], [ymin, ymax]], cmap='viridis'
    )
    axes[1].set_xlabel('Redshift (z)')
    axes[1].set_ylabel('Dispersion Measure (DM_ext)')
    axes[1].set_title('Mock sightlines')
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].grid(True)

    # use the same color limits for both plots
    vmin = min(H0.min(), H1.min())
    vmax = max(H0.max(), H1.max())
    im0.set_clim(vmin, vmax)
    im1.set_clim(vmin, vmax)

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label('Counts')
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label('Counts')

    plt.tight_layout()
    plt.show()
    
def compare_z_ra_dec():
    '''
    compare the RA and DEC distributions of sampled grid points vs mock sightlines
    '''
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # RA comparison
    axes[0].hist(los_df['ra'].values, bins=50, alpha=0.5, label='LOS ', color='blue', density=True)
    axes[0].hist(gal_df['ral'].values, bins=50, alpha=0.5, label='GAL ', color='orange', density=True)
    axes[0].set_xlabel('Right Ascension (degrees)')
    axes[0].set_ylabel('Normalized Counts')
    axes[0].set_title('RA Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True)

    # DEC comparison
    axes[1].hist(los_df['dec'].values, bins=50, alpha=0.5, label='LOS ', color='blue', density=True)
    axes[1].hist(gal_df['decl'].values, bins=50, alpha=0.5, label='GAL ', color='orange', density=True)
    axes[1].set_xlabel('Declination (degrees)')
    axes[1].set_ylabel('Normalized Counts')
    axes[1].set_title('DEC Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True)

    # z comparison as third subplot
    axes[2].hist(los_df['z'].values, bins=50, alpha=0.5, label='LOS', color='blue', density=True)
    axes[2].hist(z_samp, bins=50, alpha=0.5, label='Sampled Grid Points', color='orange', density=True)
    axes[2].hist(gal_df['z_obs'].values, bins=50, alpha=0.5, label='GAL', color='green', density=True)
    axes[2].set_xlabel('Redshift (z)')
    axes[2].set_ylabel('Normalized Counts')
    axes[2].set_title('Redshift Distribution Comparison')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


    def plot_relative_distributions():
        """
        Plot three subplots:
          - relative RA distribution: (los_ra - gal_ra) / gal_ra  (handles RA wrap)
          - relative DEC distribution: (los_dec - gal_dec) / gal_dec
          - z differences: step hist of (los_z - z_samp), (gal_z - z_samp), (los_z - gal_z)
        """
        # extract arrays and ensure same length
        ra_los = los_df['ra'].values
        ra_gal = gal_df['ral'].values
        dec_los = los_df['dec'].values
        dec_gal = gal_df['decl'].values
        z_los = los_df['z'].values
        z_gal = gal_df['z_obs'].values

        # align lengths defensively
        n = min(len(ra_los), len(ra_gal), len(dec_los), len(dec_gal), len(z_los), len(z_gal), len(z_samp))
        ra_los = ra_los[:n]; ra_gal = ra_gal[:n]
        dec_los = dec_los[:n]; dec_gal = dec_gal[:n]
        z_los = z_los[:n]; z_gal = z_gal[:n]; z_samp_local = z_samp[:n]

        eps = 1e-12

        # RA: compute minimal angular difference to avoid wrap issues, then relative
        delta_ra = ((ra_los - ra_gal + 180.0) % 360.0) - 180.0
        rel_ra = delta_ra / (ra_gal + eps)

        # DEC: simple relative difference
        delta_dec = dec_los - dec_gal
        rel_dec = delta_dec / (dec_gal + eps)

        # z differences for step histograms
        diff_los_zsamp = z_los - z_samp_local
        diff_gal_zsamp = z_gal - z_samp_local
        diff_los_gal = z_los - z_gal

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # RA relative histogram
        axes[0].hist(rel_ra, bins=50, density=True, color='C0', alpha=0.8, range=[-1.5,1.5])
        axes[0].set_xlabel('(RA_los - RA_gal) / RA_gal')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Relative RA distribution')
        # enforce symmetric x-limits: choose smallest between 0.2 and matplotlib auto-limit (by absolute value)
        auto_xmin, auto_xmax = axes[0].get_xlim()
        auto_max_abs = max(abs(auto_xmin), abs(auto_xmax))
        limit = min(0.7, auto_max_abs)
        axes[0].set_xlim(-limit, limit)
        axes[0].grid(True)

        # DEC relative histogram
        axes[1].hist(rel_dec, bins=50, density=True, color='C1', alpha=0.8)
        axes[1].set_xlabel('(DEC_los - DEC_gal) / DEC_gal')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Relative DEC distribution')
        # enforce symmetric x-limits: choose smallest between 0.2 and matplotlib auto-limit (by absolute value)
        auto_xmin, auto_xmax = axes[1].get_xlim()
        auto_max_abs = max(abs(auto_xmin), abs(auto_xmax))
        limit = min(0.2, auto_max_abs)
        axes[1].set_xlim(-limit, limit)
        axes[1].grid(True)

        # Z differences: three overlapping step histograms
        bins_z = 60
        axes[2].hist(diff_los_zsamp, bins=bins_z, histtype='step', density=True, label='los - z_samp', color='C2', linewidth=1.5)
        axes[2].hist(diff_gal_zsamp, bins=bins_z, histtype='step', density=True, label='gal - z_samp', color='C3', linewidth=1.5)
        axes[2].hist(diff_los_gal, bins=bins_z, histtype='step', density=True, label='los - gal', color='k', linewidth=1.5)
        axes[2].set_xlabel('Delta z')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Redshift difference distributions')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()


    # call the function to produce the figure
    plot_relative_distributions()




def find_gals(z_lim, select):
    '''
    find galaxies in gal_df within z+-0.02 and dec range for each sampled point
    '''
    n_samples = len(z_samp)
    gal_indices_list = []

    for i in range(n_samples):
        resample = True  # initialize 
        zcut = z_lim  # initial redshift window
        while resample:
            z_low = z_samp[i] - zcut 
            z_high = z_samp[i] + zcut 
            dec_low = low_dec_edges[i]
            dec_high = high_dec_edges[i]

            mask = (
                (gal_df['z_obs'] >= z_low) &
                (gal_df['z_obs'] <= z_high) &
                (gal_df['decl'] >= dec_low) &
                (gal_df['decl'] <= dec_high)
            )

            dataframe = gal_df[mask]

            if len(dataframe) == 0:
                zcut += 0.01  # expand search window
            else:
                resample = False  # found galaxies, exit loop
        z_low = z_samp[i] - zcut - z_lim
        z_high = z_samp[i] + zcut + z_lim
        dec_low = low_dec_edges[i]
        dec_high = high_dec_edges[i]

        mask = (
            (gal_df['z_obs'] >= z_low) &
            (gal_df['z_obs'] <= z_high) &
            (gal_df['decl'] >= dec_low) &
            (gal_df['decl'] <= dec_high)
        )

        dataframe = gal_df[mask]

        if select == 'SFR':
            # probabilities based on SFR
            sfr_values = dataframe['obssfr'].values
            probs = sfr_values / np.sum(sfr_values) 

        elif select == 'StellarMass':
            # probabilities based on Stellar Mass
            mass_values = dataframe['obssm'].values
            probs = mass_values / np.sum(mass_values)

        else:
            raise ValueError("select must be either 'SFR' or 'StellarMass'")


        gal_indices = dataframe.index.tolist()
        gal_index = np.random.choice(gal_indices, p=probs)
        gal_indices_list.append(gal_index)

    return gal_indices_list

def plot_sfr_sm_with_z():
    '''
    plot obssfr and obssm vs z_obs for galaxies in gal_df
    '''
    
    # prepare masks for stellar mass and SFR (require positive values for log)
    mask_sm = gal_df['obssm'].notna() & gal_df['z_obs'].notna() & (gal_df['obssm'] > 0)
    z_sm = gal_df.loc[mask_sm, 'z_obs'].values
    sm_raw = gal_df.loc[mask_sm, 'obssm'].values
    log_sm = np.log10(sm_raw)

    mask_sfr = gal_df['obssfr'].notna() & gal_df['z_obs'].notna() & (gal_df['obssfr'] > 0)
    z_sfr = gal_df.loc[mask_sfr, 'z_obs'].values
    sfr_raw = gal_df.loc[mask_sfr, 'obssfr'].values
    log_sfr = np.log10(sfr_raw)

    z_obs = gal_df['z_obs'].values

    # Ranges that avoid extreme outliers (use 0..99.5 percentile for y-axes)
    z_min_plot, z_max_plot = 0.0, np.nanpercentile(z_obs, 99.5)
    sfr_min, sfr_max = np.nanpercentile(log_sfr, 0.2), np.nanpercentile(log_sfr, 99.5)
    sm_min, sm_max = np.nanpercentile(log_sm, 0.2), np.nanpercentile(log_sm, 99.5)

    bins = [80, 80]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: log10(SFR) vs z_obs
    H0 = axes[0].hist2d(z_sfr, log_sfr, bins=bins, range=[[z_min_plot, z_max_plot], [sfr_min, sfr_max]], cmap='viridis')
    axes[0].set_xlabel('z_obs')
    axes[0].set_ylabel('log10(obssfr)')
    axes[0].set_title('log10(Observed SFR) vs z_obs')
    axes[0].grid(True)
    cbar0 = fig.colorbar(H0[3], ax=axes[0])
    cbar0.set_label('Counts')

    # Right: log10(Stellar mass) vs z_obs
    H1 = axes[1].hist2d(z_sm, log_sm, bins=bins, range=[[z_min_plot, z_max_plot], [sm_min, sm_max]], cmap='viridis')
    axes[1].set_xlabel('z_obs')
    axes[1].set_ylabel('log10(obssm)')
    axes[1].set_title('log10(Observed Stellar Mass) vs z_obs')
    axes[1].grid(True)
    cbar1 = fig.colorbar(H1[3], ax=axes[1])
    cbar1.set_label('Counts')

    plt.tight_layout()
    plt.show()

def plot_sfr_sm_dist():
    '''
    plot distributions of obssfr and obssm for galaxies in gal_df
    '''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # SFR distribution
    sfr_values = gal_df['obssfr'].values
    sfr_values = sfr_values[sfr_values > 0]  # filter out non-positive values for log scale
    log_sfr = np.log10(sfr_values)
    axes[0].hist(log_sfr, bins=50, color='blue', alpha=0.7)
    axes[0].set_xlabel('log10(obssfr)')
    axes[0].set_ylabel('Counts')
    axes[0].set_title('Distribution of Observed SFR')
    axes[0].grid(True)

    # Stellar Mass distribution
    sm_values = gal_df['obssm'].values
    sm_values = sm_values[sm_values > 0]  # filter out non-positive values for log scale
    log_sm = np.log10(sm_values)
    axes[1].hist(log_sm, bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('log10(obssm)')
    axes[1].set_ylabel('Counts')
    axes[1].set_title('Distribution of Observed Stellar Mass')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# --------------------------- RUN ------------------------------------------ 

test = False  # set to True for testing with smaller datasets (science runs with False)
z_max = 1.25  # maximum redshift where to integrate LOS DMs from 
z_min = 0.032  # minimum redshift -- from the galaxy catalog
z_lim = 0.01  # redshift window for finding galaxies around sampled z

num_samples = 4000  # number of sample galaxies to draw from the KDE

gal_base = 'SFR'  # 'SFR' or 'StellarMass' -- whether to weight galaxy selection by SFR or Stellar Mass

# get the pzdm grids for CHIME
all_rates, zrange, dmrange = get_grid() # the CHIME p(z, DM) grids for all declination bins.  Shape (NDECBINS, z, DM)

# sample from the CHIME p(z, DM) grids
inds = sampling(all_rates, num_samples)  # get flat indices of sampled (dec_bin, z, DM) grid cells

# convert flat indices into (dec_bin, z_idx, dm_idx)
dec_idx, z_idx, dm_idx = np.unravel_index(inds, all_rates.shape)


# produce z_samp and dm_samp as grid indices 
z_samp = zrange[z_idx]
dm_samp = dmrange[dm_idx]

# ----------- TEST  to avoid dec_idx < 5 -------------
# ensure dec_idx is a numpy array of ints
dec_idx = np.array(dec_idx, dtype=int)
# replace elements smaller than 4 with either 4 or 5 (then clamp to valid max index)
mask = dec_idx < 4
if np.any(mask):
    choices = np.random.choice([4, 5], size=mask.sum())
    dec_idx[mask] = choices
# ----------------------------------------------------

low_dec_edges = declinations[dec_idx]
high_dec_edges = declinations[dec_idx + 1]



# get host DM from lognormal distribution and extragalactic DM
lognorm_host_dm = get_lognorm_hosts(z_samp)
cosmic_dm = dm_samp - lognorm_host_dm  
#while np.any(cosmic_dm < 0):
#    # resample host DMs where cosmic DM is negative
#    neg_mask = cosmic_dm < 0
#    num_resample = np.sum(neg_mask)
#    new_lognorm_dm = get_lognorm_hosts(z_samp[neg_mask])
#    lognorm_host_dm[neg_mask] = new_lognorm_dm
#    cosmic_dm[neg_mask] = dm_samp[neg_mask] - new_lognorm_dm 


# get galaxy catalog and LOS DMs
gal_df = get_gals(gals_file, test=test) # galaxy catalog 

#plot_sfr_sm_with_z()


# find the galaxy at z+-z_lim, and dec range for each sampled point
gal_indxs = find_gals(z_lim, gal_base)
gal_df = gal_df.iloc[gal_indxs].reset_index(drop=True)

# now find the LOS that match the dec, ra, z of the galaxy and cosmic DM of each sampled point
los_df = get_los(los_file,  test=test) # cosmic DM sightlines from Konietzka et al. 2025
los_indices = match_los_to_gal()
los_df = los_df.iloc[los_indices].reset_index(drop=True)

# find the indexes in los_df that match the declination range, the z and cosmic DM of each sampled point
#los_indices = find_los(z_samp, cosmic_dm)
#los_df = los_df.iloc[los_indices].reset_index(drop=True)
# now grab the closest galaxy in the catalog
#gal_indices = match_gal_to_los(los_df, gal_df, z_samp)
#gal_df = gal_df.iloc[gal_indices].reset_index(drop=True)


# plot the sampled z and DM values
#plot_zdm_sample(z_samp, dm_samp)


# Mock sightline

z_mock = gal_df.z_obs.values
ra_mock = gal_df.ral.values
dec_mock = gal_df.decl.values
DM_extragalactic = los_df.DM_cosmic.values + lognorm_host_dm  # extragalactic DM for each LOS

plot_2dhists()
compare_z_ra_dec()
plot_sfr_sm_dist()


jkajds

# add MW ISM DM and total DM
MW_dm = get_MW_DM(ra_mock, dec_mock)  # get MW ISM DM for each LOS
MW_dm = np.array(MW_dm)
MW_halo_DM = 50.0  # constant MW halo DM contribution in pc cm^-3
DM_mock = DM_extragalactic + MW_dm + MW_halo_DM  # total DM for each LOS


mock_df = {
    'z_gal': z_mock,
    'ra_gal': ra_mock,
    'dec_gal': dec_mock,
    'DM_cosmic': los_df.DM_cosmic.values,
    'DM_MW_ISM': MW_dm,
    'DM_host': lognorm_host_dm,
    'DM_MW_halo': np.full(len(z_mock), MW_halo_DM),
    'DM_extragalactic': DM_extragalactic,
    'DM_total': DM_mock
}

mock_df = pd.DataFrame(mock_df)

# Save results to CSV
#output_file = 'CHIME_LOS_results.csv'
#mock_df.to_csv(output_file, index=False)
#print(f"Results saved to {output_file}")