"""
Directory contains code to do a convolution. Also contains the a class to convert the padding
choices from cv2.filter2D to numpy.pad.
"""

from sliding.convolution.padding import Padding, BorderType
from sliding.convolution.convolution import Convolution
