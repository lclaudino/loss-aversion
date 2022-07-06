# loss-aversion

This repo has the code for the paper: "Self-reported anxiety arousal can modulate loss aversion in pathological anxiety"

## Setting up Python and R environments

1. Install Anaconda 3 and use the loss_aversion.yml file included to recreate the environment:

```
conda env create -f loss_aversion.yml 
```

Then activate that environment:
```
conda activate loss_aversion
```


2. Install R version 3.4.1.

## To generate paper figures and data in tables -- will use provided pre-fit models
```
python loss_aversion.py
```

## To run models from scratch -- will re-fit models IF no model files are present (see source)

Using the L-BFGS-B solver (main)
```
python fit_models.py L-BFGS-B LA1

python fit_models.py L-BFGS-B LA2
```
Using the Nelder-Mead solver (supplemental)
```
python fit_models.py nelder-mead LA1

python fit_models.py nelder-mead LA2
```
Note: the source code assumes a multi-threaded environment with 50+ threads available. If that is not your setting, you will have to make changes to the code.

## To run the R part of the analysis

From your R command line, enter:

```
source 'LA.r'
```

