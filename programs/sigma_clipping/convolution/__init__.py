"""
Directory contains functions to compute a convolution. Also contains the a class to convert the
padding choices from cv2.filter2D to numpy.pad.
"""

from programs.sigma_clipping.convolution.padding import Padding, BorderType
from programs.sigma_clipping.convolution.convolution import Convolution
