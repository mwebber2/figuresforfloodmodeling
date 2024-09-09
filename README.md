# Figures for Flood Modeling Paper
Collection of Python scripts to recreate the figures used in recent paper, including data that supports the figures.

Citation for paper: Webber et al. (in prep) Addressing Local Parameter Uncertainty for Climate Resilient Urban Stormwater Modeling and Decision Making

## Dependencies
```
numpy >= 1.21.4  
pandas >= 1.5.1  
matplotlib >= 3.4.3  
```
## File Descriptions

```PCSWMM_stormsfromraingauges.xlsx```
- Each tab represents a different 1km2 grid cell with unique rainfall information
- Rows represent all events for Mar 2022 to Oct 2022
- Columns represent different characteristics of each event
- NB 4 events were dropped from the end of the record so the last observed storm used for analysis was 1 Oct 2022

```PCSWMM1D_parametersExp1.xlsx```
- Each tab represents results from a different calibration experiment: the basecase/prior model, the calibration for the most upstream location for August storms with limits on parameters, calibration for the most upstream location for all storms with limits on parameters
- For the Mar-Oct_withlimits tab that is used for figure creation, the observed and the modeled flow are recorded for each storm for the baseline/prior model, for the model calibrated to ten storms, and for the model calibrated for 1 storm
- Columns are repeated to facilitate two sets of RMSE calculations
  
```PCSWMM_multipanelscatterplot.xlsx```
- Each tab represents results for a different monitoring location
- For each monitoring location, the modeled flow is recorded for the observations, the basecase/prior model, the model calibrated to this location, the model calibrated to the most upstream location, and the model calibrated to the most downstream location.

```PCSWMM1D_RMSEresults.xlsx```
- Each tab represents decision analysis calculations and results for a different scenario: the prior model, each of the monitoring locations, combinations of monitoring locations (all6 or all4). The last three tabs summarize the information from the previous tabs for 1) all storms 2) calibration storms only (10), 3) validation storms only (44).
