"""
Code to convert the padding choices from cv2.filter2D to numpy.pad.
"""
from __future__ import annotations

# IMPORTs alias
import numpy as np

# TYPE ANNOTATIONs
from typing import Literal, Any, cast
import numpy.typing as npt
type BorderType = Literal['reflect', 'constant', 'replicate'] | None

# API public
__all__ = ['BorderType', 'Padding']



class Padding[Data: npt.NDArray[np.floating[Any]]]:
    """
    To add padding to data according to the border type.
    The border type follows the cv2.filter2D border naming convention.
    """

    def __init__(
            self,
            data: Data,
            kernel: tuple[int, ...],
            borders: BorderType = 'reflect',
        ) -> None:
        """
        Adds padding to the given data according to the border type.
        The border type follows the cv2.filter2D border naming convention. The padding is added
        using np.pad.
        To get the padded data, use the 'padded' property.

        Args:
            data (Data): the data to pad.
            kernel (tuple[int, ...]): the kernel size used for the convolution.
            borders (BorderType, optional): the border type to use for padding.
                Defaults to 'reflect'.
        """

        self._data = data
        self._kernel = kernel
        self._borders = borders

        # RUN
        self._padded_data = self._add_padding()

    @property
    def padded(self) -> Data:
        """
        The padded data using np.pad and the borders choice.

        Returns:
            Data: the padded data.
        """
        return self._padded_data

    def _get_padding(self) -> dict:
        """
        Gives a dictionary containing the np.pad padding choices equivalent to the border
        information.

        Raises:
            ValueError: if the border type name is not recognised.

        Returns:
            dict: the dictionary containing the padding choices.
        """

        if self._borders is None:
            # ADAPTATIVE borders
            result = {
                'mode': 'constant',
                'constant_values': np.nan,
            }
        elif self._borders == 'reflect':
            result = {
                'mode': 'symmetric',
                'reflect': 'even',
            }
        elif self._borders == 'constant':
            result = {
                'mode': 'constant',
                'constant_values': 0.,
            }
        elif self._borders == 'replicate':
            result = {'mode': 'edge'}
        else:
            raise ValueError(f"Unknown border type: {self._borders}")
        return result

    def _add_padding(self) -> Data:
        """
        To add padding to the given data according to the border type.

        Returns:
            np.ndarray: the padded data.
        """

        # WIDTH padding
        pad = tuple((k // 2, k // 2) for k in self._kernel)

        # MODE np.pad
        padding_params = self._get_padding()
        padding_mode = padding_params['mode']
        padding_constant_values = padding_params.get('constant_values', 0)
        padding_reflect_type = padding_params.get('reflect', 'even')

        if padding_mode == 'constant':
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
                constant_values=padding_constant_values,
            )
        elif padding_mode == 'symmetric':
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
                reflect_type=cast(Literal['even'], padding_reflect_type),
            )
        else:
            padded = np.pad(
                array=self._data,
                pad_width=pad,
                mode=cast(Literal['edge'], padding_mode),
            )
        return cast(Data, padded)
