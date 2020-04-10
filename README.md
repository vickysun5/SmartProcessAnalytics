# SmartProcessAnalytics
Smart Process Analytics for predictive modeling

This repository contains Smart Process Analytics (SPA) softawre for predictive modeling associated with the paper 'Smart Process Analytics' by Weike Sun and Richard D. Braatz.

The softawre is performed in Python, the `Smart_Process_Analytics` file is the main function to run the Smart Process Analytics which provides the default version for the predictive modeling and will provide the final model with model evaluation for you. All the other files that are needed to run `Smart_Process_Analytics` is stored in `Code-SPA`. To run it on your computer, simply download the `Smart_Process_Analytics` and all the other files in `Code-SPA` and put them in the same folder.

If you want to reset other hyperparameters that are set as default values in the `Smart_Process_Analytics` file or use other functionalities (e.g., model fitting for multiple times series data set using one model), you can use the data interrogation, model construction and residual analysis by yourself. One example is provided in the file `Example 1` in Folder `Example` which uses different methods directly called from the `cv_final` file using the data from the 3D printer example in the original paper.



The major files under the `Smart_Process_Analytics` are:
1. `dataset_property_new`: functions for data interrogation.
2. `cv_final`/`cv_final_onestd`: model construction using different cross-validation strageries (or cross-validation with one standard error rule) for models in SPA.
3. `IC`: model construction using information criteria for dynamic models.
4. `regression_models/nonlinear_regression_other`: basic linear/nonlinear and DALVEN regression models in SPA.
5. `timeseries_regression_RNN`: RNN model (including training/testing for single/multiple training sets).
6. `timeseries_regression_matlab`: MATLAB SS model (including training/testing for single/multiple training sets).
7. `timeseries_regression_ADAPTx`: ADAPTx SS-CVA model (including training/testing for single/multiple training sets.

The final result is stored in `selected_model` (for the name that the software picked or the user spicified) and `fitting_result`.


Note: For linear dynamic model, MATLAB is required to implement the software and is called through matlab.engine (https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html). If the user has ADAPTx, it can also be implemented for linear state-space model through SPA. The url of ADAPTx pacakge will be asked when implementing SPA.
The acepack package (https://cran.r-project.org/web/packages/acepack/acepack.pdf) in R is required to conduct nonlinearity assessment. The SPLS package (https://cran.r-project.org/web/packages/spls/index.html) in R is required to do sparese PLS regression. Please modify the url for package location in file `ace_R` and `SPLS`. rpy2 is required to call these R packages (https://rpy2.readthedocs.io/en/latest/).



Please contact Richard Braatz at braatz@mit.edu for any inquiry. 
