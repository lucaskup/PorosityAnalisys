# Hyperspectral Data is a Viable Proxy for Porosity Estimation of Carbonate Rocks
This repository provides the code, data and the results used to estimate the porosity through spectral data. 

# Table of contents 

- [Requirements](#requirements) 
- [How to cite](#how-to-cite) 
- [Credits](#credits) 
- [License](#license) 

## Requirements
    pandas
    numpy
    scikit-learn
    seaborn
    matplotlib
    scipy

### Folder Structure

    In the code folder you will find .py files with the implementation used in this study.
    .
    ├── code
        ├──compare_models.py              # Variance corrected t-test to evaluate the residuals;
        ├──correlation_matrix.py          # Auxiliary function to plot a correlation matrix;
        ├──descriptive_stats.py           # notebook that calculate descriptive stats listed in the paper;
        ├──feature_selection.py           # Feature selection phase of the methodology described in the paper;
        ├──learning_curve.py              # Plots the learning curve, Figure 8 in the manuscript;
        ├──mnf_rotation_test.py           # tests mnf pre-processing;
        ├──model_training.py              # Model training and hyperparameter tuning considering the feature selected dataset;
        ├──plot_correlation_matrix.py     # Printing of the correlation matrix and correlation plot;
        ├──correlation_matrix.py          # Plots a simple scatter plot between single selected features and target.
    ├── data                              # Dataset used in the research
    ├── results                           # contains plots, saved models and metrics obtained as results.
    └── README.md
    
## How to cite

If you find our work useful in your research please consider citing [our paper](https://doi.org/10.1080/08120099.2022.2046636):

Kupssinskü, L., Guimarães, T., Cardoso, M., Bachi, L., Zanotta, D., Estilon de Souza, I., … Gonzaga, L. (2022). Hyperspectral Data as a Proxy for Porosity Estimation of Carbonate Rocks. Australian Journal of Earth Sciences. doi:10.1080/08120099.2022.2046636

```
@article{kupssinsku_AJES_2022,
  author = {Kupssinskü, Lucas and 
            Guimarães, Taina and 
            Cardoso, Milena and 
            Bachi, Leonardo and 
            Zanotta, Daniel and 
            Estilon de Souza, Italos and 
            Falcao, Alexandre and 
            Velloso, Raquel and 
            Cazarin, Caroline and 
            Veronez, Maurício and 
            Gonzaga Jr, Luiz},
  title = {Hyperspectral Data as a Proxy for Porosity 
           Estimation of Carbonate Rocks},
  journal = {Australian Journal of Earth Sciences},
  year = {2022},
  doi = {10.1080/08120099.2022.2046636}
}
```

## Credits
This work is credited to the [Vizlab | X-Reality and GeoInformatics Lab](http://www.vizlab.unisinos.br/) and the following authors and developers: [Lucas Silveira Kupssinskü](https://www.researchgate.net/profile/Lucas_Kupssinskue).

## License
``` 
MIT Licence (https://mit-license.org/) 
``` 
