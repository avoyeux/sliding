"""
Directory contains numba optimized functions to compute a sliding weighted median where in the
input data can contain NaN values.
"""

from sliding.mode.numba_functions.sliding_median import sliding_weighted_median_nd
