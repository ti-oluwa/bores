import typing

import numba  # type: ignore[import-untyped]
import numpy as np
from numba.extending import overload  # type: ignore[import-untyped]

__all__ = [
    "apply_mask",
    "clip",
    "clip_scalar",
    "get_mask",
    "is_array",
    "max_",
    "min_",
]


@numba.vectorize(cache=True)
def clip(val, min_, max_):
    return np.maximum(np.minimum(val, max_), min_)


@numba.njit(cache=True)
def clip_scalar(value: float, min_val: float, max_val: float) -> float:
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    return value


@typing.overload
def is_array(x: np.ndarray) -> typing.TypeGuard[np.typing.NDArray]: ...


@typing.overload
def is_array(x: typing.Any) -> typing.TypeGuard[np.typing.NDArray]: ...


@numba.njit(cache=True)
def is_array(x: typing.Any) -> bool:
    return hasattr(x, "shape") and isinstance(x.shape, tuple)


@numba.njit(cache=True)
def _apply_mask_2d(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to a 2D array where mask is True (in-place).

    :param arr: 2D array to modify
    :param mask: 2D boolean mask with same shape as arr
    :param values: scalar or 2D array of values to assign where mask is True
    """
    nx, ny = arr.shape

    for i in range(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                arr[i, j] = values[i, j]


@numba.njit(cache=True)
def _apply_mask_3d(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to a 3D array where mask is True (in-place).

    :param arr: 3D array to modify
    :param mask: 3D boolean mask with same shape as arr
    :param values: scalar or 3D array of values to assign where mask is True
    """
    nx, ny, nz = arr.shape
    for i in range(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    arr[i, j, k] = values[i, j, k]


@numba.njit(cache=True)
def _apply_mask_nd(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Apply values (scalar or array) to an N-dimensional array where mask is True (in-place).

    :param arr: N-dimensional array to modify
    :param mask: N-dimensional boolean mask with same shape as arr
    :param values: scalar or N-dimensional array of values to assign where mask is True
    """
    for idx in np.ndindex(arr.shape):
        if mask[idx]:
            arr[idx] = values[idx]


@numba.njit(cache=True)
def apply_mask(
    arr: np.typing.NDArray, mask: np.typing.NDArray, values: np.typing.NDArray
) -> None:
    """
    Dispatcher to apply scalar or array values to an array where mask is True.

    :param arr: Array to modify (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as arr
    :param values: scalar or array of values to assign where mask is True
    """
    ndim = arr.ndim
    if ndim == 2:
        _apply_mask_2d(arr, mask, values)
    elif ndim == 3:
        _apply_mask_3d(arr, mask, values)
    else:
        _apply_mask_nd(arr, mask, values)


@numba.njit(cache=True)
def _get_mask_2d(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new 2D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: 2D input array
    :param mask: 2D boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: 2D array with masked values applied
    """
    nx, ny = arr.shape
    out = np.empty_like(arr)
    for i in range(nx):  # type: ignore
        for j in range(ny):
            if mask[i, j]:
                out[i, j] = arr[i, j]
            else:
                out[i, j] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_3d(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new 3D array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: 3D input array
    :param mask: 3D boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: 3D array with masked values applied
    """
    nx, ny, nz = arr.shape
    out = np.empty_like(arr)
    for i in range(nx):  # type: ignore
        for j in range(ny):
            for k in range(nz):
                if mask[i, j, k]:
                    out[i, j, k] = arr[i, j, k]
                else:
                    out[i, j, k] = fill_value
    return out


@numba.njit(cache=True)
def _get_mask_nd(arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float):
    """
    Return a new N-dimensional array where values are kept if mask is True, otherwise replaced with fill_value.

    :param arr: N-dimensional input array
    :param mask: N-dimensional boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: N-dimensional array with masked values applied
    """
    out = np.empty_like(arr)
    for idx in np.ndindex(arr.shape):
        if mask[idx]:
            out[idx] = arr[idx]
        else:
            out[idx] = fill_value
    return out


@numba.njit(cache=True)
def get_mask(
    arr: np.typing.NDArray, mask: np.typing.NDArray, fill_value: float = np.nan
):
    """
    Dispatcher to return a masked copy of an array.

    :param arr: Input array (2D, 3D, or N-dimensional)
    :param mask: Boolean mask with same shape as arr
    :param fill_value: Scalar value to fill where mask is False
    :return: Array with masked values applied
    """
    ndim = arr.ndim
    if ndim == 2:
        return _get_mask_2d(arr, mask, fill_value)
    elif ndim == 3:
        return _get_mask_3d(arr, mask, fill_value)
    return _get_mask_nd(arr, mask, fill_value)


# When used in pure-python, this called
def min_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.min(x)


def max_(x) -> np.floating[typing.Any]:
    if isinstance(x, float):
        return x  # type: ignore[return-value]
    return np.max(x)


# In numba context, these overloads are used
@overload(min_)
def min_overload(x):
    # SCALAR CASE
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    # ARRAY CASE
    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.min(x)

        return impl


@overload(max_)
def max_overload(x):
    if isinstance(x, numba.types.Number):

        def impl(x):
            return x

        return impl

    if isinstance(x, numba.types.Array):

        def impl(x):
            return np.max(x)

        return impl
