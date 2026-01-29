"""
Directory contains code to perform different implementation of the sigma clipping algorithm.
"""

from programs.sigma_clipping.convolution import BorderType, Convolution, Padding
from programs.sigma_clipping.sliding_mode import SlidingMean, SlidingMedian
from programs.sigma_clipping.standard_deviation import SlidingStandardDeviation
from programs.sigma_clipping.old_sigma_clipping import sigma_clip
from programs.sigma_clipping.sigma_clipping import SigmaClipping
