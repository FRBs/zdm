import numpy as np

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