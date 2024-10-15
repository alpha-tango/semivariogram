Getting Started:
- Clone the repo
- `cd` into the repo directory
- If you're using a virtual environment, pip install the requirements in requirements.txt.
- If you're using Conda, you may only need to install DuckDB: python-duckdb on Conda forge.

If you want to produce some initial charts to look at the Berea dataset, run:
`python3 driver_berea.py raw_histogram` 
It will show charts and save them in the images/ directory.

If you want to produce a raw semivariogram, you have options for binning strategies.
You can run:
`python3 driver_berea.py raw_semivariogram`
...and you will get a raw semivariogram with equal-points-per-bin (displayed and saved
in images/).

You can run:
`python3 driver_berea.py raw_semivariogram --bin_width=5`
...and you will get a raw semivariogram with bins 5 mm wide. (Or whatever other number
you specify).

You can run:
`python3 driver_berea.py raw_semivariogram -bx 5 -bx 20 -bx 100 -bx 250`
and each of those numbers will be used as the division point (in lag mm) for the bins.
(Or whatever other numbers you specify, without good error handling).

If you want to produce a semivariogram, you can run:
`python3 driver_berea.py semivariogram --range=40 --model=spherical`
You can also combine this with the binning options above (as written, this will use
equal-points-per-bin). This will display and save the model semivariogram. Range
is something you choose.

Currently supported models: spherical, gaussian, exponential, isotonic. See `--help` option
for additional information on options.
