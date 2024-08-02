This code is the calculation of V/Vmax for [paper link]

It runs with four sets of data:
- All localised FRBs
- All FRBs used z(DM)
- Unbiased localised FRBs (max z = 0.7) ("UnbiasedLocalised")
- Unlocalised FRBs setting a minimum z ("Minz")


###### calc_vvmax.py #######

This will generate V and Vmax values for FRBs, and store them in files
in a defined output directory (e.g. v2Output)
the various combinations are:
- Localised or Macquart DM
- Min redshift z
- Max redshift z
- value of spectral index alpha
- value of population evolution scaling

Output in "[label]Output/"

##### calc_lf_errors.py #####

Runs a MC rountine to calculate errors in the luminosity function
Inputs from above, outputs in another directory
("[label]LumData")


##### plotting! ######

Plots for the paper are:


- make_paper_unbiased_lum_plot
Plots Minz and UnbiasedLocalised data
Shows effects of systematics (not strong! Interesting...)
