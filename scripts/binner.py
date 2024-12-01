"""
Create binned data for modeling the semivariogram.

The binners require a an input pair_df which must have the following columns:
    - `h` is the lag distance between that pair of points
    - `semivariance` is the semivariance between that pair of points

The bins() function for each class returns a dataframe with the following columns:
- `bin`: a float the represents the max `h` that corresponds to the data in the bin
- `h`: the average distance in that bin
- `semivariance`: the average semivariance in that bin
- `n`: the count of data points in that bin
"""
import duckdb

import math

class RawBinner:
    """
    Don't do anything lol.
    """

    def __init__(self):
        pass

    def bins(self, pair_df):
        no_bins = """
        SELECT
        h AS bin,
        h,
        semivariance,
        STDDEV_POP(semivariance) AS stddev,
        1 AS n
        FROM
        pair_df
        """
        return duckdb.sql(no_bins).df()


class CustomBinner:
    """
    Pass in a custom list of bin maxes
    """

    def __init__(self, bin_maxes:list[float]):
        assert len(bin_maxes) > 0  # need at least one bin
        self.bin_maxes = bin_maxes

    def bins(self, pair_df):

        # mapping function for binning
        def binner(x):

            # want the first bin that is greater than or equal to the distance
            # first, make sure bins are sorted
            self.bin_maxes.sort()

            for m in self.bin_maxes:
                if m >= x:
                    return m
            
            # if we reach this point,
            # the user didn't specify a bin large enough for the biggest lag
            # so just bin all data above their largest bin together
            return pair_df['h'].max()

        # map each pair to a bin max
        pair_df['bin'] = pair_df['h'].map(binner)
        print(pair_df.head(20))
        
        # use DuckDB query to calculate per-bin average lag and semivariance
        binner = f"""
        SELECT
            bin,
            AVG(h) AS h,
            AVG(semivariance) AS semivariance,
            STDDEV_POP(semivariance) AS stddev,
            COUNT() AS n
        FROM pair_df
        GROUP BY bin
        ORDER BY 1
        """
        
        # turn query results into dataframe
        return duckdb.sql(binner).df()

        

class EqualWidthBinner:
    """
    Each bin has an equal width (an equal span of lag distance).
    """

    def __init__(self, bin_width):
        # bin width has to be more than zero
        assert bin_width > 0
        self.bin_width = bin_width

    def bins(self, pair_df):
        """
        Use duckDB query to bin the data.
        """
        binner = f"""
        WITH bins AS (
            SELECT
                h AS lag_distance,
                semivariance,
                {self.bin_width} AS bin_width,
                CEIL(h / {self.bin_width}) AS bin
            FROM pair_df
            ORDER BY h ASC
        )

        SELECT
            bin,
            AVG(lag_distance) AS h,
            AVG(semivariance) AS semivariance,
            STDDEV_POP(semivariance) AS stddev,
            COUNT() AS n
        FROM bins
        GROUP BY bin
        ORDER BY 1
        """

        # turn query results into dataframe
        return duckdb.sql(binner).df()



class EqualPointBinner:
    """
    Each bin has an equal number of points.
    Takes in a dataframe of pairs which must have the following columns:
    - `h` is the lag distance between that pair of points
    - `semivariance` is the semivariance between that pair of points
    """

    def __init__(self, points_per_bin=None):
        self.points_per_bin = points_per_bin

    def sqrt_bins(self, pair_df):
        """
        Calculate number of bins using square-route method shown in class
        """
        pair_count = pair_df['h'].count()
        return math.floor(pair_count ** (1/2))

    def bins(self, pair_df):
        """
        Use DuckDB to bin the data
        """
        if not self.points_per_bin:
            self.points_per_bin = self.sqrt_bins(pair_df=pair_df)

        binner = f"""
            WITH bins AS (
                SELECT
                    h AS lag_distance,
                    semivariance AS semivariance,
                    CEIL(ROW_NUMBER() OVER (ORDER BY h ASC) / {self.points_per_bin}) * {self.points_per_bin} AS bin_id
                FROM 
                    pair_df
            )

            SELECT
                MAX(lag_distance) AS bin,
                AVG(lag_distance) AS h,
                AVG(semivariance) AS semivariance,
                STDDEV_POP(semivariance) AS stddev,
                COUNT() AS n
            FROM
                bins
            GROUP BY bin_id
            ORDER BY 1
            """

        # turn query results into dataframe
        return duckdb.sql(binner).df()

