"""
Config settings for berea dataset
"""

######################################
# image settings and functions
######################################

IM_TAG = 'berea_test'

def custom_raw_plots(raw_df):
    import matplotlib.pyplot as plt

    # Plot location vs permeability
    fig, ax = plt.subplots()
    ax.scatter(raw_df['x'], raw_df['primary'])
    ax.set_title("Permeability by x location")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Permeability (md)")
    plt.show()
    fig.savefig('images/berea_permeability_by_x.png')

    fig, ax = plt.subplots()
    ax.scatter(raw_df['y'], raw_df['primary'])
    ax.set_title("Permeability by y location")
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("Permeability (md)")
    plt.show()
    fig.savefig('images/berea_permeability_by_y.png')


######################################
# Preparing raw data for pairwise-df
######################################

def raw_data():
    """
    Return a dataframe with the following columns:
    `id`: an id used for row counting in creating pairs.
    `x`: x location.
    `y`: y location.
    `primary`: the primary parameter to look at. 
    `secondary`: (optional) the secondary parameter.
    """
    import duckdb

    berea_select = """
        SELECT
        row_number() OVER () AS id,
        x_mm AS x,
        y_mm AS y,
        permeability_md AS primary,
        water_resisitivity_ohmm AS secondary
        FROM read_csv('data/berea_subset2.csv', normalize_names=True)
        ORDER BY x, y ASC
        """
    return duckdb.sql(berea_select).df()


######################################
# Binning data
######################################

# set a list of max bins for custom bins
BIN_MAXES = [3,5,10,15,20,25,30,40,500]



