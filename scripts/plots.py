"""
All the code in one place to make the plt plots with correct formatting
and labels.
Some column names are hard-coded, lots of fixes needed.
"""
import matplotlib.pyplot as plt

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
                        raw_df, imname):
        self.a = a  # range
        self.omega = omega  # sill
        self.model_name = model_name
        self.model_x = model_lag
        self.model_y = model_semivariance
        self.raw_df = raw_df
        self.fig, self.ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        self.imname = imname

    def labels(self):
        """
        Label the axes appropriately.
        TODO: add in unit for x-axis
        """

        # Main Semivariogram
        title_str = f'Semivariogram: {self.model_name} model'
        x_str = 'Lag Distance (h)'
        y_str = 'Semivariance'

        self.ax[0].set_title(title_str)
        self.ax[0].set_xlabel(x_str)
        self.ax[0].set_ylabel(y_str)

        # N display
        self.ax[1].set_xlabel("Lag Distance (h)")
        self.ax[1].set_ylabel("Points per Bin")

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
        self.ax[0].text(0.75, 0.25, textstr, transform=self.ax[0].transAxes, fontsize=10,
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



