import numpy as np
from pyproj import Geod


###################################
# RAW / INDIVIDUAL STATS
###################################

def geographic_distance(long_x_1, long_x_2, lat_y_1, lat_y_2, radians=False):
    """
    Geographic (on the earth) distance in kilometers given vectors of lat longs.
    radians=False assumes data is given in degrees (normal lat/long)
    rather than in radians.
    """

    # set earth projection to use
    g = Geod(ellps="WGS84")

    # g.inv returns a tuple of forward azimuths, back azimuths, and distances in meters
    # we only care about the distances, which is the 3rd item (index 2) in the tuple
    inv = g.inv(lons1=long_x_1, lats1=lat_y_1, lons2=long_x_2, lats2=lat_y_2, radians=radians)

    # return the distance portion only in kilometers rather than meters
    return inv[2] / 1000.0


def euclidean_distance_2d(x1, x2, y1, y2):
    """
    Euclidean distance function for 2 (x, y) positions.
    """
    return ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

def raw_semivariance(z, v):
    """
    Calculate raw semivariance AKA individual semivariance.
    The semivariance for one pair of values.
    The vector semivariance function below calculates for vectors.
    """
    return (z - v)**2 / 2



###################################
# VECTOR STATS
###################################

def covariance(x,y):
  """
  Calculate the covariance of two vectors
  """
  assert len(x) == len(y)  # vectors must be the same length for this to work

  n = len(x)
  mean_x = mean(x)
  mean_y = mean(y)

  return np.sum((x - mean_x)*(y- mean_y)) / n

def mean(x):
  """
  Calculate the mean of a vector
  """
  n = len(x)
  return sum(x) / n

def pearson_correlation(x,y):
  """
  Calculate the Pearson correlation coefficient
  between two vectors.
  """
  assert len(x) == len(y)  # vectors must be the same length for this to work

  n = len(x)
  stddev_x = std_dev(x)
  stddev_y = std_dev(y)

  return covariance(x,y) / (stddev_x * stddev_y)

def std_dev(v):
  """
  Calculate the standard deviation of a vector
  """
  return np.sqrt(variance(v))

def semivariance(x, y):
  """
  Calculate the semivariance of two vectors
  """
  assert len(x) == len(y)  # vectors must be the same length for this to work
  n = len(x)
  return np.sum((x - y)**2) / (2*n)


def variance(v):
  """
  Calculate the variance of a vector
  """
  mean = np.mean(v)
  n = len(v)
  return np.sum((v - mean)**2) / n

########################################
# CROSS STATS
########################################

def cross_covariance(primary, secondary):
    """
    Calculate the cross-covariance of two vectors.
    It's the same as the normal covariance, this wrapper just clarifies
    primary vs. secondary.

    The argument `primary` ought to be the primary stat, not lagged (near).
    The argument `secondary` ought to be the secondary stat, lagged (far).
    """
    return covariance(primary, secondary)

def cross_pearson_correlation(primary, secondary):
    """
    Calculate the cross-correlation coefficient.
    It's the same as the normal Pearson correlation, so this is just a wrapper
    to clarify primary vs secondary.

    The argument `primary` ought to be the primary stat, not lagged (near).
    The argument `secondary` ought to the secondary stat, lagged.
    """
    return pearson_correlation(primary, secondary)

def cross_semivariance(primary_near, primary_far, secondary_near, secondary_far):
    """
    Calculate cross-semivariance.
    This one is NOT the same as the normal semivariance.
    The argument `primary_near` ought to be the primary stat, not lagged.
    The argument `primary_far` ought to be the primary stat, lagged.
    The argument `secondary_near` ought to be the secondary stat, not lagged.
    The argument `secondary_far` ought to be the secondary stat, lagged.
    """

    # all the vectors must be the same length
    assert len(primary_near) == len(primary_far)
    assert len(primary_near) == len(secondary_near)
    assert len(primary_near) == len(secondary_far)

    n = len(primary_near)

    return np.sum(
          (primary_near - primary_far) \
          * (secondary_near - secondary_far)
          ) / (2 * n)