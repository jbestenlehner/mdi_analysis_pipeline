# mdi_analysis_pipeline
Spectroscopic analysis pipeline using model de-idealisation to derive physical parameters of stars using synthetic spectra computed with stellar atmosphere codes.

The code is presented in Bestenlehner et al. (2024): "Spectroscopic analysis of hot, massive stars in large spectroscopic surveys with de-idealized models", MNRAS, 528, 6735.

Link to journal: https://academic.oup.com/mnras/article/528/4/6735/7592034

DOI: https://doi.org/10.1093/mnras/stae298

## Installation

To install, clone or download the zip file containing the code into your python or virtual environment Lib folder, where you want to execute the pipeline. Even though a setup.py file might be present, I am still in the process to package the code. For now contact me, if you have any questions or issues running the pipeline. 

## Using the code

1. Create folders for your observational data (obs_dat) and decomposed model grid (mod_dat).  
2. The grid of models can be decomposed using the example script "create_decomposition_matrix_from_grid_of_stellar-atmospheres.py"
3. Update the path and fits file extensions in "many_stars_run_shared-memory.py" and create an input csv-file (input.csv).
4. The "input.csv" file should contain the following columns: "target,filename,vsini,vmac,RV" with "target" a unique target identify, "filename" filename of spectra (e.g. target.fits), vsini (projected rotational velocity) and vmac (macro-turbulent velocity) are currently not used, and RV (radial velocity).
5. To run the code execute the following command in the command line "python many_stars_run_shared-memory.py".

## Attribution

If you have found this code useful, then please cite it as [Bestenlehner et al., 2024 (MNRAS, 528, 6735)](https://ui.adsabs.harvard.edu/abs/2024MNRAS.528.6735B/abstract).

