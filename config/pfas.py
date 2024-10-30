"""
Config settings for berea dataset
"""

######################################
# image settings and functions
######################################

IM_TAG = 'pfas'

def custom_raw_plots(raw_df):
    import matplotlib.pyplot as plt

    # Plot location vs concentration
    fig, ax = plt.subplots()
    ax.scatter(raw_df['x'], raw_df['primary'])
    ax.set_title("PFAA concentration by latitude")
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_lat.png')

    fig, ax = plt.subplots()
    ax.scatter(raw_df['y'], raw_df['primary'])
    ax.set_title("PFAA concentration by longitude")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_long.png')

    # Plot concentration vs date sampled
    fig, ax = plt.subplots()
    ax.scatter(raw_df['sampling_date'], raw_df['primary'])
    ax.set_title("PFAA concentration by sampling date")
    ax.set_xlabel("Sampling Date")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_date.png')

    # Plot concentration vs rain
    fig, ax = plt.subplots()
    ax.scatter(raw_df['rain'], raw_df['primary'])
    ax.set_title("PFAA concentration by previous rain")
    ax.set_xlabel("Rain")
    ax.set_ylabel("Concentration (ng/kg dry soil weight)")
    plt.show()
    fig.savefig(f'images/{IM_TAG}_concentration_by_date.png')



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

    pfas_select = """
        SELECT
        ROW_NUMBER() OVER () AS id,
        CONCAT(data.sample_id, sub_id) AS sample_id,
        property_latitude AS x,
        property_longitude AS y,
        sum_PFAA AS primary,
        CASE WHEN data.rain IS NOT NULL THEN TRUE
        ELSE FALSE END AS rain,
        sampling_date,
        property_name
        FROM read_csv('data/VT_pfas_data_clean.csv', normalize_names=True) AS data
        INNER JOIN read_csv('data/VT_pfas_ID.csv', normalize_names=True) AS sample_id
        ON data.sample_id = sample_id.sample_id
        """
    return duckdb.sql(pfas_select).df()


######################################
# Binning data
######################################

# set a list of max bins for custom bins
# BIN_MAXES = [3,5,10,15,20,25,30,40,500]



