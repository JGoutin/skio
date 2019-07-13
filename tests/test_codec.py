"""Tests for skio/codec.py"""
from skio import intdecode, intencode
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal


IDAT = np.array(((0, 1, 2), (3, 4, 5)))  # Integer data
FDAT = np.array(((0.0, 0.5, 1.0), (1.5, 2.0, 2.5)))  # Float data
FDAT_NEG = np.array(((-2.0, -1.0, 0.0), (1.0, 2.0, 3.0)))  # Data with value <0
FDAT_NAN = np.array(((0.0, 0.5, 1.0), (np.nan, 2.0, 2.5)))  # Data with nan
MDAT = np.array(((False, False, False), (True, False, False)))  # Nan Mask


def test_intdecode_invalid():
    """'intdecode' function: 'invalid' argument"""
    # None
    assert_equal(intdecode(IDAT, 0.5, invalid=None), FDAT)
    # None: Test mask
    assert not intdecode(IDAT, 0.5, invalid=None, masked=True).mask.any()
    # Single
    assert_equal(intdecode(IDAT, 0.5, invalid=3), FDAT_NAN)
    # Single: Test mask
    assert_equal(intdecode(IDAT, 0.5, invalid=3, masked=True).mask, MDAT)
    # Tuple
    assert_equal(intdecode(IDAT, 0.5, invalid=(1, 4)),
                 np.array(((np.nan, np.nan, 1.0), (1.5, np.nan, np.nan))))
    # Tuple: Test mask
    assert_equal(intdecode(IDAT, 0.5, invalid=(1, 4), masked=True).mask,
                 np.array(((True, True, False), (False, True, True))))


def test_intdecode_dtype():
    """'intdecode' function: 'dtype' argument"""
    # set dtype
    assert intdecode(IDAT, 0.5, dtype=np.float32).dtype == np.float32
    # Not a floating type
    with pytest.raises(ValueError) as excinfo:
        intdecode(IDAT, 0.5, dtype=np.int32)
    assert 'dtype must be a floating data type' in str(excinfo.value)


def test_intencode_invalidvalue():
    """'intencode' function: 'invalidvalue' argument"""
    # False : With no invalid data
    data, factor = intencode(FDAT, np.int16, invalidvalue=False)
    assert_equal(data, np.array(((0, 6553, 13107), (19660, 26214, 32767))))
    assert_almost_equal(factor, 7.6296273689992981e-05)
    # False : With invalid data
    data = intencode(FDAT_NAN, np.int16, invalidvalue=False)[0]
    assert_equal(data.mask, MDAT)
    assert_equal(data.data, np.array(((0, 6553, 13107), (0, 26214, 32767))))
    # None : With signed int
    assert_equal(intencode(FDAT_NAN, np.int16, invalidvalue=None)[0],
                 np.array(((0, 6553, 13107), (-32768, 26214, 32767))))
    # None : With unsigned int
    assert_equal(intencode(FDAT_NAN, np.uint16, invalidvalue=None)[0],
                 np.array(((0, 13107, 26214), (65535, 52427, 65534))))
    # Specified value
    assert_equal(intencode(FDAT_NAN, np.int16, invalidvalue=-1)[0],
                 np.array(((0, 6553, 13107), (-1, 26214, 32767))))


def test_intencode_rangeminmax():
    """'intencode' function: 'rangemin' & 'rangemax' arguments"""
    # Negative and positive min and max
    assert_equal(intencode(FDAT_NEG, np.int16, rangemin=-100,
                 rangemax=100)[0],
                 np.array(((-67, -33, 0), (33, 67, 100))))
    # Negative and positive min and max with inverted data
    assert_equal(intencode(FDAT_NEG * -1, np.int16, rangemin=-100,
                           rangemax=100)[0], np.array(((67, 33, 0),
                                                       (-33, -67, -100))))
    # Positive min and max
    assert_equal(intencode(FDAT_NEG, np.int16, rangemin=100, rangemax=200)[0],
                 np.array(((100, 120, 140), (160, 180, 200))))
    # Negative min and max
    assert_equal(intencode(FDAT_NEG, np.int16, rangemin=-200,
                 rangemax=-100)[0],
                 np.array(((-200, -180, -160), (-140, -120, -100))))
    # Too larges values
    assert_equal(intencode(FDAT_NEG, np.int8, rangemin=-256, rangemax=256)[0],
                 np.array(((-85, -42, 0), (42, 85, 127))))


def test_intencode_keepsign():
    """'intencode' function: 'keepsign' argument"""
    # Keep
    assert_equal(intencode(FDAT_NEG, np.int16, keepsign=True)[0],
                 np.array(((-21845, -10922, 0), (10922, 21845, 32767))))
    # Don't keep
    assert_equal(intencode(FDAT_NEG, np.int16, keepsign=False)[0],
                 np.array(((0, 6553, 13107), (19660, 26214, 32767))))
    # Keep but unsigned
    assert_equal(intencode(FDAT_NEG, np.uint16, keepsign=True)[0],
                 np.array(((0, 13107, 26214), (39321, 52428, 65535))))


def test_intencode_int_inv_max():
    """'intencode' function: 'intfactor', 'maxfactor', 'invfactor' arguments"""
    # Float
    assert intencode(FDAT, np.uint8, rangemax=1, intfactor=False)[1] == 2.5
    # Float inverted
    assert_almost_equal(intencode(FDAT, np.uint8, rangemax=1, invfactor=True,
                                  intfactor=False)[1], 0.4)
    # Float inverted maxed
    assert intencode(FDAT, np.uint8, maxfactor=10, invfactor=True,
                     intfactor=False)[1] == 10
    # Float inverted maxed with max = int max
    assert intencode(FDAT * 1e-5, np.uint8, maxfactor=-1, invfactor=True,
                     intfactor=False)[1] == 255
    # Float maxed
    assert_almost_equal(intencode(FDAT, np.uint8, maxfactor=1e-5,
                                  intfactor=False)[1], 1e-5)
    # Integer
    assert intencode(FDAT, np.uint8, rangemax=1, intfactor=True)[1] == 3
    # Integer inverted
    assert intencode(FDAT, np.uint8, rangemax=1, invfactor=True,
                     intfactor=True)[1] == 1
    # Integer inverted maxed
    assert intencode(FDAT, np.uint8, maxfactor=10, invfactor=True,
                     intfactor=True)[1] == 10
    # Integer maxed
    assert intencode(FDAT, np.uint8, maxfactor=1, intfactor=True)[1] == 1


def test_intencode_forcefactor():
    """'intencode' function: 'forcefactor' argument"""
    assert intencode(FDAT, np.uint8, forcefactor=42)[1] == 42


def test_intencode_intround():
    """'intencode' function: 'intround' argument"""
    # Round
    assert_equal(intencode(FDAT_NEG, np.int16, intround=True)[0],
                 np.array(((-21845, -10922, 0), (10922, 21845, 32767))))
    # Don't round
    assert_equal(intencode(FDAT_NEG, np.int16, intround=False)[0],
                 np.array(((-21844, -10922, 0), (10922, 21844, 32767))))


def test_intencode_inttype():
    """'intencode' function: character code as int dtype"""
    # Character code
    assert_equal(intencode(FDAT, 'h')[0],
                 np.array(((0, 6553, 13107), (19660, 26214, 32767))))
