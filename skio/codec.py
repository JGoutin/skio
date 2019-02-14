"""data encoding utilities"""
# TODO:
# - Better docstring intros for int codecs.

import numpy as np
from collections.abc import Iterable


def intdecode(data, scalefactor, dtype=np.float64, invalid=None, masked=False):
    """
    Decode float datas from scaled integers datas.

    Parameters
    ----------
    data : numpy.ndarray
        Data to decode.
    scalefactor : float or int
        scalefactor to apply.
    dtype : floating data-type, optional
        Floating Data type of the returned array.
    invalid : int or tuple of int
        Replace values that count as invalid by a mask if "masked" is True
        or by np.nan if "masked" is False.
        If int, replace equal values.
        If tuple like (low limit, high limit), replace outside values.
    masked : bool
        If True, return a masked array with invalid values masked. If False,
        return a ndarray with nans invalid values.

    Return
    -------
    out : numpy.ndarray/numpy.ma.MaskedArray
        Decoded datas.
    """
    if not (issubclass(dtype, np.floating) or issubclass(dtype, float)):
        raise ValueError('dtype must be a floating data type')

    # Remove invalid values
    if invalid is None:
        # No invalid values
        if masked:
            filtered = np.ma.array(data, dtype=dtype, copy=True)
        else:
            filtered = np.array(data, dtype=dtype)

    elif isinstance(invalid, Iterable):
        # low/high limits invalid values
        imin, imax = invalid
        filtered = np.where((data > imin) & (data < imax), data, np.nan)
    else:
        # Single invalid value
        filtered = np.where(data != invalid, data, np.nan)

    # Mask nans
    if invalid and masked:
        filtered = np.ma.fix_invalid(filtered, copy=False)

    # Re-scale data
    return filtered * scalefactor


def intencode(data, inttype, invalidvalue=False, rangemin=None,
              rangemax=None, keepsign=True, intfactor=False, invfactor=False,
              maxfactor=0, forcefactor=0, intround=True):
    """
    Encode float data to integers with optimal scale factor.

    Parameters
    ----------
    data: numpy.ndarray
        Data to scale. Invalid values must be masked.
    inttype: type or str
        Integer type or numpy dtype character code.
    invalidvalue: int, optional
        Value for replacing invalid data (nan, inf, masked...).
        If "None", use minimum value for signed formats and maximum value for
        unsigned formats.
        If "False", return numpy.ma.MaskedArray if invalid data are found.
    rangemin: int, optional
        Minimum value after scaling. If "None", use default format range min.
    rangemax: int, optional
        Maximum value after scaling. If "None", use default format range max.
    keepsign: bool, optional
        If "True", keep sign while scaling on signed formats. If "False"
        scale the full amplitude on the full integer range.
    intfactor: bool, optional
        If "True", the scale factor will be int, else it will be float.
    invfactor: bool, optional
        If "True", return 1/scale factor. If "False", return directly the
        scale factor.
    maxfactor: int or float, optional
        If not "0", the scale factor is calculed for not be greater than this
        value. If "-1", the max value used is the max limit for the selected
        intformat.
    forcefactor: int or float
        If not "0", force the use of the specified factor.
    intround: bool, optional
        If "True", round data when converting to int, else truncate.

    Return
    -------
    out : tuple (numpy.ndarray/numpy.ma.MaskedArray, float/int)
        (Encoded datas, scale factor)
    """

    # Convert character code
    if isinstance(inttype, str):
        inttype = np.dtype(inttype)

    # Get integer type limits
    info = np.iinfo(inttype)
    intmin, intmax = info.min, info.max

    # Initialise data as temporary float with invalid data masked
    intdata = np.ma.masked_invalid(data.astype(np.float64), copy=False)

    # Set invalid value and keepsign flag
    if intmin == 0:
        if intdata.min() < 0:
            # Negatives values not compatibles with uint
            keepsign = False
        if invalidvalue is None:
            # Invalid value = max in uint
            invalidvalue = intmax
    else:
        if invalidvalue is None:
            # Invalid value = min in signed int
            invalidvalue = intmin

    # Set scaling range
    if rangemin is None:
        if invalidvalue != intmin or invalidvalue is False:
            rangemin = intmin
        else:
            rangemin = intmin + 1
    elif rangemin < intmin:
        # Fix if incompatible range
        rangemin = intmin

    if rangemax is None:
        if invalidvalue != intmax:
            rangemax = intmax
        else:
            rangemax = intmax - 1
    elif rangemax > intmax:
        # Fix if incompatible range
        rangemax = intmax

    if not keepsign and rangemin < 0:
        # Fix if incompatible range
        rangemin = 0
    elif rangemin >= 0 or rangemax <= 0:
        # Don't keep sign if not on both sides of 0
        keepsign = False

    # Set Scale factor
    if forcefactor:
        # User selected scale factor
        factor = forcefactor
    else:
        # Calculate best scale factor
        if keepsign:
            minval = abs(intdata.min())
            maxval = abs(intdata.max())
            minrng = abs(rangemin)
            maxrng = abs(rangemax)
            if minval / minrng > maxval / maxrng:
                maxdata = minval
                maxrange = minrng
            else:
                maxdata = maxval
                maxrange = maxrng
        else:
            # Scale to full range
            maxdata = intdata.max() - intdata.min()
            maxrange = rangemax - rangemin
        factor = maxdata / maxrange

    # If reversed factor
    if invfactor:
        factor = 1 / factor
        intrev = -1
    else:
        intrev = 1

    # If integer factor only
    if intfactor:
        ifactor = int(factor)
        if factor != ifactor:
            # Apply reversed factor correction
            ifactor += intrev
        factor = ifactor

    # Check factor max limit
    if maxfactor:
        if maxfactor == -1:
            maxfactor = intmax
        if factor > maxfactor:
            factor = maxfactor

    # Check factor sign
    if factor < 0:
        factor = abs(factor)

    # Scale data
    if invfactor:
        intdata *= factor
    else:
        intdata /= factor

    # If loose sign, move min to zero
    if rangemin >= 0 or not keepsign:
        intdata += rangemin - intdata.min()

    # Round data
    if intround:
        intdata = np.around(intdata)

    # Cast to integer type
    intdata = intdata.astype(inttype)

    # Replace invalid values
    if invalidvalue:
        intdata = intdata.filled(invalidvalue)
    elif not np.ma.is_masked(intdata):
        # Return masked array only if invalid values found and no replacement
        intdata = np.array(intdata, copy=False)

    return intdata, factor
