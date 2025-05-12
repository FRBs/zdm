#Module for Tables for the Mannings+21 HST paper
# Imports
import numpy as np
import os, sys
import pandas


from zdm.MC_sample import loading
from zdm import survey
from zdm import parameters
from zdm import misc_functions
from zdm import io
from zdm import iteration as it

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import analy_H0_I

def mktab_model_params(outfile='tab_model_params.tex', sub=False):
    # Load up 
    #base_survey='CRAFT_CRACO_MC_base'
    #survey, grid = loading.survey_and_grid(survey_name=base_survey)
    isurvey, grid = analy_H0_I.craco_mc_survey_grid()

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\begin{minipage}{170mm} \n')
    tbfil.write('\\caption{Fiducial Set of Model Parameters. ')
    tbfil.write('Parameters labelled with a * are re-fit as part of this work. \n') 
    tbfil.write('\\label{tab:param}}\n')
    tbfil.write('\\begin{tabular}{cccl}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Parameter & Fiducial Value & Unit & Description \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    add_star = ['lmean', 'lsigma', 'sfr_n', 'lEmax', 'alpha', 'gamma', 'H0']

    for key, item in grid.state.params.items():
        # Ones to skip
        if key in ['ISM', 'luminosity_function', 'Omega_k',
                   'fix_Omega_b_h2', 'alpha_method', 'lC',
                   'source_evolution', 'Wthresh', 'Wmethod', 'Wbins', 'Wscale',
                   'Sfpower', 'Sfnorm']: 
            continue
        # Include these dicts
        if item not in ['energy', 'host', 'width', 'MW', 'IGM',
                        'FRBdemo', 'cosmo', 'scat']:
            continue
        # Name
        if key in add_star:
            add_not = '^*'
        else:
            add_not = ''
        try:
            slin = f'${getattr(grid.state, item).meta(key)["Notation"]}{add_not}$'
            # Value
            if key in ['Omega_lambda', 'Omega_b_h2']:
                slin += f'& {getattr(getattr(grid.state, item),key):.5f}'
            else:
                slin += f'& {getattr(getattr(grid.state, item),key)}'
            # Unit
            slin += f'& {getattr(grid.state, item).meta(key)["unit"]}'
            # Descirption
            slin += f'& {getattr(grid.state, item).meta(key)["help"]}'
        except:
            print(f"Failed on key={key}, item={item}.  Fix it!")
            embed(header='69 of tables')
        # Add extras
        if key == 'DMhalo':
            tbfil.write('DM$_{\\rm ISM} & NE2001 & \\dmunits & DM for the Milky Way Interstellar Medium \\\\ \n')
        elif key == 'F':
            tbfil.write('$f_d(z=0)$ & 0.844 & & fraction of baryons that are diffuse and ionized at $z=0$ \\\\ \n')
        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


def mktab_frbs(outfile='tab_frbs.tex', sub=False):

    state = parameters.State()
    zDMgrid, zvals,dmvals = misc_functions.get_zdm_grid(
        state, new=True, plot=False, method='analytic')
    # Load up the surveys
    names = ['CRAFT/ICS', 'CRAFT/FE', 'CRAFT/ICS892', 'PKS/Mb']
    names = ['CRAFT/ICS']

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\begin{minipage}{170mm} \n')
    tbfil.write('\\caption{Fast Radio Bursts Analyzed\\label{tab:frbs}}\n')
    tbfil.write('\\begin{tabular}{cccccccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Name & Survey & DM & $\\nu$ & SNR & RA & Dec & $z$ \n')
    tbfil.write('& & & (pc cm$^{-3}$) & (MHz) & & (J2000) & (J2000) & \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    # Loop on survey
    for name in names:
        isurvey = survey.load_survey(name, state, dmvals)
        #embed(header='111 of tables')
        # Loop on FRBs
        for ss in range(isurvey.NFRB):
            slin = f'{isurvey.frbs["ID"][ss]}'
            slin += f'& {name}'
            slin += f'& {isurvey.frbs["DM"][ss]}'
            slin += f'& {isurvey.frbs["FBAR"][ss]}'
            slin += f'& {isurvey.frbs["SNR"][ss]}'
            slin += f'& {isurvey.frbs["XRa"][ss]}'
            slin += f'& {isurvey.frbs["XDec"][ss]}'
            slin += f'& {isurvey.frbs["Z"][ss]}'

            # Write
            tbfil.write(slin)
            tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))


def mktab_MC(outfile='tab_MC.tex', sub=False):

    if sub:
        outfile = 'tab_MC_sub.tex'

    '''
    # Load 
    input_dict=io.process_jfile('../Analysis/Cubes/craco_sfr_Emax_cube.json')

    # Deconstruct the input_dict
    state_dict, cube_dict, vparam_dict = it.parse_input_dict(input_dict)

    ############## Initialise ##############
    isurvey, grid = loading.survey_and_grid(
        state_dict=state_dict,
        survey_name='CRACO_alpha1_Planck18_Gamma',
        NFRB=100,
        iFRB=100)
    '''

    isurvey, grid = analy_H0_I.craco_mc_survey_grid()

    # Open
    tbfil = open(outfile, 'w')

    # Header
    #tbfil.write('\\clearpage\n')
    tbfil.write('\\begin{table*}\n')
    tbfil.write('\\centering\n')
    tbfil.write('\\begin{minipage}{170mm} \n')
    tbfil.write('\\caption{CRACO Monte Carlo\\label{tab:MC}}\n')
    tbfil.write('\\begin{tabular}{cccc}\n')
    tbfil.write('\\hline \n')
    tbfil.write('Index & DM & \\snr & $z$ \n')
    tbfil.write('\\\\ \n')
    tbfil.write(' & (\\dmunits) \n')
    tbfil.write('\\\\ \n')
    tbfil.write('\\hline \n')

    for ss in range(isurvey.NFRB):
        if sub and ss > 15:
            continue
        slin = f'{ss}'
        slin += f'& {isurvey.frbs["DM"][ss]}'
        slin += f'& {isurvey.frbs["SNR"][ss]}'
        slin += f'& {isurvey.frbs["Z"][ss]}'

        # Write
        tbfil.write(slin)
        tbfil.write('\\\\ \n')

    # End
    tbfil.write('\\hline \n')
    tbfil.write('\\end{tabular} \n')
    tbfil.write('\\end{minipage} \n')
    #tbfil.write('{$^a$}Rest-frame value.  Error is dominated by uncertainty in $n_e$.\\\\ \n')
    #tbfil.write('{$^b$}Assumes $\\nu=1$GHz, $n_e = 4 \\times 10^{-3} \\cm{-3}$, $z_{\\rm DLA} = 1$, $z_{\\rm source} = 2$.\\\\ \n')
    tbfil.write('\\end{table*} \n')

    tbfil.close()

    print('Wrote {:s}'.format(outfile))

# Command line execution
if __name__ == '__main__':

    mktab_model_params()
    #mktab_frbs()
    #mktab_MC()
    #mktab_MC(sub=True)
