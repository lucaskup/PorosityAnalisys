# Hyperspectral Data is a Viable Proxy for Porosity Estimation of Carbonate Rocks
This repository provides the code, data and the results used to estimate the porosity through spectral data. 

# Table of contents 

- [Requirements](#requirements) 
- [Usage](#usage) 
- [How to cite](#how-to-cite) 
- [Credits](#credits) 
- [License](#license) 

## Requirements
    pandas
    numpy
    scikit-learn

### Folder Structure

    In the code folder you will find .py files with the implementation used in this study.
    .
    ├── code
        ├──compare_models.py              # Variance corrected t-test to evaluate the residuals;
        ├──correlation_matrix.py          # Auxiliary function to plot a correlation matrix;
        ├──feature_selection.py           # Feature selection phase of the methodology described in the paper;
        ├──model_training.py              # Model training and hyperparameter tuning considering the feature selected dataset;
        ├──plot_correlation_matrix.py     # Printing of the correlation matrix and correlation plot .
    ├── data                              # Dataset used in the research
    ├── results                           # contains plots, saved models and metrics obtained as results.
    └── README.md
    
## How to cite

Yet to be published. Manuscript submited to the Australian Journal of Earth Sciences.

## Credits
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://www.vizlab.unisinos.br/) and the following authors and developers: [Lucas Silveira Kupssinskü](https://www.researchgate.net/profile/Lucas_Kupssinskue).

## License
``` 
MIT Licence (https://mit-license.org/) 
``` 
