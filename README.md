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


### A typical top-level directory layout

    .
    ├── build                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── src                     # Source files (alternatively `lib` or `app`)
    ├── test                    # Automated tests (alternatively `spec` or `tests`)
    ├── tools                   # Tools and utilities
    ├── LICENSE
    └── README.md
    In the code folder you will find .py files with the implementation used in this study.
    .
    ├── code
        ├── *compare_models.py* # Implements a variance corrected t-test to evaluate if the difference in the models is statistically significant;
        ├──*correlation_matrix.py* # Auxiliary function to plot a correlation matrix;
        ├──*feature_selection.py*  # Implements the feature selection phase of the methodology described in the paper;
        ├──*model_training.py* # Implements model training and hyperparameter tuning considering the feature selected dataset;
        ├──*plot_correlation_matrix.py*  # Implements the printing of the correlation matrix and correlation plot that are found in the draft submited to Computers and Geosciences.
    ├── data  # contains the dataset used in the research
    ├── results # contains plots, graphs, csv files and metrics obtained by executing the scripts under \code directory

## How to cite

Yet to be published. Manuscript submited to Elsevier Computers & Geosciences.

## Credits
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://www.vizlab.unisinos.br/) and the following authors and developers: [Lucas Silveira Kupssinskü](https://www.researchgate.net/profile/Lucas_Kupssinskue).

## License
``` 
MIT Licence (https://mit-license.org/) 
``` 
