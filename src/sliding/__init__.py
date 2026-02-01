"""
Directory contains code to perform n-dimensional sliding mean, median, standard deviation,
and sigma clipping on numpy arrays (with/without NaNs) given a kernel (with/without weights).
"""

from sliding.convolution import BorderType, Convolution, Padding
from sliding.mode import SlidingMean, SlidingMedian
from sliding.standard_deviation import SlidingStandardDeviation
from sliding.sigma_clipping import SlidingSigmaClipping
