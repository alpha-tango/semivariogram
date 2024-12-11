# Getting Started
- Clone the repo
- `cd` into the repo directory
- If you're using a virtual environment, pip install the requirements in requirements.txt.
- If you're using Conda, you may only need to install DuckDB and PyProj: python-duckdb and pyproj on Conda forge, respectively
- You will need the PFAA dataset in the `data/` folder. (not publicly available)

# Semivariograms
If you want to produce some initial charts to look at dataset, run:
`python3 driver.py pfaa raw_histogram` 
It will show plots and save them in the `images/` directory.

If you want to produce an experimental semivariogram, run:
`python3 driver.py pfaa raw_semivariogram`
It will show plots and save them in the `images/` directory. 
Binning methods for the experimental data are specified in the config file.

If you want to produce a semivariogram, you can run:
`python3 driver.py pfaa semivariogram`
Binning methods and model settings are specified in the config file.
It will show plots and save them in the `images/` directory. 

See `--help` option for additional information on options.

# Kriging
- Run `python3 kriging.py pfaa` and then look in the `images/` directory for `pfaa_kriged_error.png` and `pfaa_kriged_values.png`
- Only Ordinary Kriging is currently supported
- Only kriging with semivariance is currently supported.
- Use the `--test` command line argument to test on a small set of target points specified in the config file.
- Use the `--validation` command line argument to run validation on a set of primary data points specified in the config file.

# Co-Kriging
- Run `python3 cokriging.py pfaa` and then look in the `images/` directory for `pfaa_cokriged_values.png` and some additional custom plots.
- Only co-kriging with covariance is currently supported.
- Only ordinary kriging is currently supported.
- Use the `--test` command line argument to test on a small set of target points specified in the config file.
- Use the `--validation` command line argument to run validation on a set of primary data points specified in the config file.

# Running with your own data
You will need to create a config file for your dataset (one for each primary variable).  You will save it in the `config/` directory. 
The actual settings you need to define are different for each task -- `config/pfaa.py` has a fairly complete set of variables and functions
defined (and commented!). A template config file is in the works. With each of the main drivers (`semivariogram.py`, `kriging.py`, and 
`cokriging.py`) you will use the name of the config file as a command line argument.
For example, if you create `config/mydata.py`, you could run `python3 kriging.py mydata` or `python3 cokriging.py mydata`.
