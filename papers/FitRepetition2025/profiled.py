PROFILED_PID = None
import cProfile
from zdm import MCMC

def profiled_calc_log_posterior(param_vals, state, params, surveys_sep, Pn=False, Pns=False, Pnr=False,
                pNreps=True, psnr=True, ptauw=False, pwb=False,
                log_halo=False, lin_host=False, ind_surveys=False, g0info=None, nz=500, ndm=1400,
                zmax=5.,dmmax=7000.,
                dopath=False, opstate=None, opt_params=None, opt_model=None):
    
    global PROFILED_PID
    pid = os.getpid()

    # If we haven't chosen a worker yet, choose this one
    if PROFILED_PID is None:
        PROFILED_PID = pid

    if pid == PROFILED_PID:
        profiler_output = f"worker_{pid}.prof"
        return cProfile.runctx(
            "calc_log_posterior(param_vals, state, params, surveys_sep, Pn, Pns, Pnr, "
            "pNreps, psnr, ptauw, pwb, log_halo, lin_host, ind_surveys, g0info, nz, ndm, zmax,dmmax, "
            "dopath, opstate, opt_params, opt_model)", 
            globals(), 
            locals(), 
            profiler_output
        )
    else:
        return MCMC.calc_log_posterior(param_vals, state, params, surveys_sep, Pn, Pns, Pnr, 
                pNreps, psnr, ptauw, pwb, log_halo, lin_host, ind_surveys, g0info, nz, ndm, zmax,dmmax,
                dopath, opstate, opt_params, opt_model)
