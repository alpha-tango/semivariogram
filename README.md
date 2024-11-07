# Getting Started
- Clone the repo
- `cd` into the repo directory
- If you're using a virtual environment, pip install the requirements in requirements.txt.
- If you're using Conda, you may only need to install DuckDB and PyProj: python-duckdb and pyproj on Conda forge, respectively

# Semivariograms: Berea test data
If you want to produce some initial charts to look at the Berea dataset, run:
`python3 driver.py berea raw_histogram` 
It will show charts and save them in the `images/` directory.

If you want to produce a raw semivariogram, you have options for binning strategies.
You can run:
`python3 driver.py berea raw_semivariogram --bin_method=equal_points`
...and you will get a raw semivariogram with equal-points-per-bin (displayed and saved
in `images/`).

You can run:
`python3 driver.py berea raw_semivariogram --bin_method=fixed_width --bin_width=5`
...and you will get a raw semivariogram with bins 5 units wide. (Or whatever other number
you specify).

You can run:
`python3 driver.py berea raw_semivariogram -bin_method=config`
...and the bin maxes listed in `config/berea.py` will be used.

If you want to produce a semivariogram, you can run:
`python3 driver.py berea semivariogram --range=40 --model=exponential --bin_method=equal_points`
You can also combine this with the binning options above (as written, this will use
equal-points-per-bin). This will display and save the model semivariogram. Range
is something you choose.

Currently supported models: exponential. See `--help` option
for additional information on options.

# Semivariograms: running with your own data
- Make a file in the `config/` directory.
- Write a `def raw_data()` function to grab your raw data and return a dataframe with the required columns: `id`, `x`, `y`, and `primary` (and however many other columns you want)
- Set the variable `IM_TAG` in that file (this will be the prefix for output images)
- See the `config/` directory for examples and other functions and settings you can add.

# Kriging: HW 6
- Run `python kriging.py hw6 --model=hw6 --range=10` and then look in the `images/` directory for `hw6_kriged_error.png` and `hw6_kriged_values.png`
- Sill and nugget are currently hardcoded, but can be altered in the `scripts/models.py` file in the `HW6Model` class.
- Only Ordinary Kriging is currently supported
