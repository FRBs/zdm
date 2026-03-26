The scripts in this folder were written by Bryce Smith,
with minor adaptations by C.W. James.

The intention is to evaluate the effect of photometric redshifts
on H0 estimation using a simple 1D scan.

The order of operation is:

1: python create_fake_surveys.py

This generates fake surveys for CRACO and MeerTRAP
For each, there are four: base, with mag limit, with photometric smearing, and with both.

2: run_H0_slice.py

This runs a slice through H0 over all surveys. Data are saved in directory H0.
All eight surveys (MeerTRAP and CRACO) are expected to be run at the same time

python run_H0_slice.py -n 101 --min=50 --max=100 -f CRACO/Smeared CRACO/zFrac CRACO/Spectroscopic CRACO/Smeared_and_zFrac MeerTRAP/Smeared MeerTRAP/zFrac MeerTRAP/Spectroscopic MeerTRAP/Smeared_and_zFrac

3: run python plot_h0_slice
generates the figure H0 scan_linear

4: run python plot_2D_grids.py
Generates plots of the 2D grids for each fake survey
