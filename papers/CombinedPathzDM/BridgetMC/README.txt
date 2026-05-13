This directory contains files for generating the MC sample of FRB hosts.

STEP 1: GEN MC FRBS (pythpn gen_mc_frbs_w_hosts.py)

    It begins by generating an MC sample of FBBs.
    That will produce plots in MC_Generation_Plots
    and the csv file craco_900_mc_sample.csv

    Because we use a very fine grid in b and w for this, the process takes O~few minutes

Step 2: ASSIGN HOSTS (python run_assign_host.py)
    
    Reads in the mc frb csv from above, and assigns a host
    galaxy to each FRB from the catalogue
    combined_HSC_DECaLs_HECATE_galaxies_hecatecut.parquet
    These FRBs are written to craco_assigned_galaxies.csv
    
    Typically takes ~1 minute for 10,000 FRBs.
    
    It may produce unassigned FRBs if there are many very faint or bright hosts.

STEP 3: WRITE FAKE SURVEY FILE (python write_fake_survey_file.py)
    
    Writes fake survey files corresponding to the MC generation.
    
    Three get generated in the directory "Surveys"
    Each has 100, 1000, and about 10,000 FRBs, respectively
    
    
STEP 4: WRITE FRB and PATH files (python write_frb_and_cand_files.py)

    We next generate candidate files, and FRB files, for use by PATH

    This produces fake PATH csv input files in CandidateFiles,
    and fake FRB json files in FRBFiles.
    
    FRBFiles takes about 1-2 seconds - it's quick.

    Writing fake candidate files takes a long time, due to having to query
    the catalogues. O~ 2 hrs. If you only want to use the first e.g. 1000
    FRBs, then one could begin an MCMC run part-way into writing
    the files.


STEP 5: TEST LIKELIHOOD EVALUATION (python test_likelihood.py)

    Creates surveys and grids for each of the three fake surveys,
    and evaluates likelihoods with each. Generally, this will be
    OK, provided that it doesn't crash!
