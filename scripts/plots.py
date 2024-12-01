"""
All the code in one place to make the plt plots with correct formatting
and labels.
Some column names are hard-coded, lots of fixes needed.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import isotonic_regression


class SampleLocations:
    """
    Plot the sample locations of raw data.
    """
    def __init__(self, imname, raw_df):
        self.fig, self.ax = plt.subplots()
        self.imname = imname
        self.raw_df = raw_df

    def labels(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(f"Location of {self.raw_df['x'].count()} samples")

    def scatter(self):
        # Plot x,y of sample locations
        self.ax.scatter(self.raw_df['x'], self.raw_df['y'])

    def _build(self):
        self.scatter()
        self.labels()

    def show_and_save(self):
        """
        Create and pop up the chart, then save.
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_sample_locations.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_sample_locations.png')


class RawPairData:
    """
    Plot the raw pair data.
    """
    def __init__(self, imname, pair_df):
        self.fig, self.ax = plt.subplots()
        self.imname = imname
        self.pair_df = pair_df

    def labels(self):
        self.ax.set_title("Semivariance by lag distance")
        self.ax.set_xlabel("Lag distance")
        self.ax.set_ylabel("Semivariance")

    def scatter(self):
        self.ax.scatter(self.pair_df['h'], self.pair_df['semivariance'], s=0.2)

    def text_box(self):
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = f"pair count = {self.pair_df['h'].count()}"

        # place a text box in upper left in axes coords
        self.ax.text(0.05, 0.95, textstr, transform=self.ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

    def _build(self):
        self.scatter()
        self.labels()
        self.text_box()

    def show_and_save(self):
        """
        Create and pop up the chart, then save. 
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_raw_pairs.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_raw_pairs.png')


class RawHistogram:
    """
    Make a histogram of the raw data.
    """
    def __init__(self, imname, pair_df):
        self.fig, self.ax = plt.subplots()
        self.imname = imname
        self.pair_df = pair_df

    def labels(self):
        self.ax.set_title("Count of sample pairs by lag distance")
        self.ax.set_xlabel("Lag Distance")
        self.ax.set_ylabel("Count of Pairs")

    def histogram(self):
        self.ax.hist(self.pair_df['h'])

    def _build(self):
        self.histogram()
        self.labels()

    def show_and_save(self):
        """
        Create and pop up the chart, then save. 
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_raw_histogram.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_raw_histogram.png')


class RawSemivariogram:
    """
    Make a raw semivariogram.
    - Subplot 0 displays a scatter of raw points, with the averaged points on top
    - Subplot 1 displays the points per bin

    `avg_df` must have columns ['h', 'semivariance', 'n'] for the binned data.
        'h': lag distance (avg per bin)
        'semivariance': semivariance (avg per bin)
        'n': count of points in that bin

    `pair_df` must have the columns ['h', 'semivariance']
    """

    def __init__(self, imname, pair_df, avg_df, bin_width=None, bin_max=None, n_bins=None, h_units=None):
        self.fig, self.ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.avg_df = avg_df
        self.pair_df = pair_df
        self.imname = imname
        self.bin_width = bin_width
        self.bin_max = bin_max
        self.n_bins = n_bins
        self.h_units = h_units

    def full_pair_scatter(self):
        """
        Plot the raw pairs
        """
        num_points = len(self.pair_df['h'])
        if num_points > 1000:
            s = 0.2
        if num_points > 100:
            s = 1
        if num_points > 10:
            s = 5
        else:
            s = 10
        self.ax[0].scatter(self.pair_df['h'], self.pair_df['semivariance'], color="lightgray", s=s)

    def bin_scatter(self):
        """
        Plot the number of points per bin.
        """
        self.ax[1].scatter(self.avg_df['h'], self.avg_df['n'])

    def avg_scatter(self):
        """
        Plot the average semivariance points
        """
        self.ax[0].scatter(self.avg_df['h'], self.avg_df['semivariance'], color='red', marker='x')

    def labels(self):
        title_str = 'Raw semivariogram'
        
        if self.bin_width: 
            title_str += f': {self.bin_width} fixed bins'
        elif self.bin_max:
            title_str += ': bin divisions at \n' + ', '.join([str(int(i)) for i in self.bin_max])
        else:
            title_str += f': equal points per bin ({self.n_bins} bins)'

        self.ax[0].set_title(title_str)
        self.ax[0].set_xlabel(f'Distance ({self.h_units})')
        self.ax[0].set_ylabel('Semivariance')

    def text_box(self):
         # add in the Ns


        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = f"pair count = {self.pair_df['h'].count()}"

        # place a text box in upper left in axes coords
        self.ax[0].text(0.05, 0.95, textstr, transform=self.ax[0].transAxes, fontsize=10,
                verticalalignment='top', bbox=props)


    def _build(self):
        self.full_pair_scatter()
        self.avg_scatter()
        self.bin_scatter()
        self.labels()
        self.text_box()

    def show_and_save(self):
        """
        Create and pop up the chart, then save. 
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_raw_semivariogram.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_raw_semivariogram.png')


class IsotonicSmooth():
    """
    Show a plot with smoothed raw pair data (via isotonic regression)
    and suggested range and sill.
    """
    def __init__(self, imname, pair_df):
        self.imname = imname
        self.pair_df = pair_df
        self.fig, self.ax = plt.subplots()
        self.sill = 0
        self.range = 0

    def isotonic_y(self):
        """
        Smooth input pair data into constraint that it's monotonically increasing.
        See docs on isotonic regression for more details
        """
        return isotonic_regression(self.pair_df['semivariance'])

    def set_sill_range(self, tolerance=0.07):
        """
        Suggest a sill and range for the data.
        """

        counter = 0
        curr_sill = 0
        curr_range = 0
        pair_count = len(self.pair_df['h'])

        for i,j in enumerate(self.isotonic_y()):
            
            if j == curr_sill:
                counter += 1
            else:
                counter = 0
                curr_range = self.pair_df['h'][i]

            if counter >= pair_count * tolerance:
                self.sill = curr_sill
                self.range = curr_range
                return
            
            curr_sill = j
        
        self.sill = curr_sill
        self.range = curr_range
        return

    def _build(self):
        self.set_sill_range()
        y = self.isotonic_y()
        self.ax.plot(self.pair_df['h'], y, label="Isotonic regression on pairs")
        self.ax.axhline(self.sill, color='black', label=f"Suggested sill: {self.sill:.2f}")
        self.ax.bar(self.range, height = y.max(), label=f"Suggested range: {self.range:.2f}", color="orange")
        self.ax.set_title("Pairs Smoothed with Isotonic Regression")
        self.ax.legend()


    def show_and_save(self):
        """
        Create and pop up the chart, then save. 
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_isotonic.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_isotonic.png')

class Semivariogram:
    """
    All the code to make a semivariogram.

    Some column names are hard-coded, so not wholly reusable at this point.

    Raw_df must have columns ['h', 'semivariance', 'n'] for the binned data.
        'h': lag distance (avg per bin)
        'semivariance': semivariance (avg per bin)
        'n': count of points in that bin
    """

    def __init__(self, a, omega, model_name, model_lag, model_semivariance, 
                        raw_df, imname, h_units):
        self.a = a  # range
        self.omega = omega  # sill
        self.model_name = model_name
        self.model_x = model_lag
        self.model_y = model_semivariance
        self.raw_df = raw_df
        self.fig, self.ax = plt.subplots(nrows=2, ncols=1, sharex=True, height_ratios=[3,1])
        self.fig.tight_layout()
        self.imname = imname
        self.h_units = h_units

    def labels(self):
        """
        Label the axes appropriately.
        TODO: add in unit for x-axis
        """

        # Main Semivariogram
        title_str = f'Semivariogram: {self.model_name} model'
        x_str = f'Distance ({self.h_units})'
        y_str = 'Semivariance'

        self.ax[0].set_title(title_str)
        self.ax[0].set_xlabel(x_str)
        self.ax[0].set_ylabel(y_str)

        # N display
        self.ax[1].set_xlabel(f"Distance ({self.h_units})")
        self.ax[1].set_ylabel("Points per Bin")
        self.ax[1].set_title("Bin Maximums")

    def textbox(self):
        """
        Display a text box with the range (a) and sill (omega)
        specified.
        """
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join((
            f"sill = {self.omega:.1f}",
            f"range = {self.a}"
            ))
        self.ax[0].text(0.75, 0.75, textstr, transform=self.ax[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    def scatter_points(self):
        """
        Add in scatterplot of points used to fit the model.
        """
        self.ax[0].scatter(
            self.raw_df['h'],
            self.raw_df['semivariance'],
            color='red',
            marker='x')

    def confidence_intervals(self):
        """
        Shaded bar for 95% confidence.
        Using equation from class:
        semivar_mean_in_bin +- 1.96 * std_dev_semivar_in_bin / sqrt(n_pairs_in_bin)
        """
        adj = 1.96 * self.raw_df['stddev'] / self.raw_df['n'] ** (1/2)
        y1 = self.raw_df['semivariance'] + adj
        y2 = self.raw_df['semivariance'] - adj

        self.ax[0].fill_between(self.raw_df['h'], y1, y2, alpha=0.2)

    def model_line(self):
        """
        Plot the modeled line.
        """
        self.ax[0].plot(self.model_x, self.model_y, color='black')

    def bin_scatter(self):
        """
        Plot the number of points per bin.
        """
        self.ax[1].scatter(self.raw_df['h'], self.raw_df['n'])
        height = max(self.raw_df['n'].values)
        for h in self.raw_df['bin']:
            self.ax[1].bar(x=h, height=height, color='lightgray')

    def range_sill(self):
        """
        Plot the range and sill.
        """
        self.ax[0].axvline(x=self.a,
                label=f'Range: {self.a}',
                color='black',
                linestyle='--')

        self.ax[0].axhline(y=self.omega,
            label=f"Sill: {self.omega}",
            color='black',
            linestyle=':')

    def _build(self):
        self.confidence_intervals()
        self.scatter_points()
        self.model_line()
        self.bin_scatter()
        self.textbox()
        self.labels()
        self.textbox()


    def show_and_save(self):
        """
        Create and pop up the chart, then save. 
        """
        self._build()
        plt.show()
        self.fig.savefig(f'images/{self.imname}_semivariogram.png')

    def save(self):
        """
        Create and save chart without displaying.
        """
        self._build()
        self.fig.savefig(f'images/{self.imname}_semivariogram.png')



