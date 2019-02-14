"""Tests for skio/system.py"""
from skio import validfilename
import pytest
import os.path


def test_validfilename():
    """'validfilename' function: with representatives invalids characters"""
    name = ' - --  a*sf/J|<IZ?E->p{}v\\64"8d:f.gfg. ... .'.format(chr(31))
    assert validfilename(name) == 'asfJIZE-pv648df.gfg'


def test_validfilename_fullpath():
    """'validfilename' function: 'fullpath' argument"""
    path = os.path.normpath('/skio/test/test_system.py')
    # Path
    assert validfilename(path, fullpath=True) == path
    # Filename
    assert validfilename(path, fullpath=False) == 'skiotesttest_system.py'


def test_validfilename_iso9660():
    """'validfilename' function: 'iso9660' argument"""
    name = '--d1d5s1--11sd--dd--'
    assert validfilename(name, iso9660=True) == 'd1d5s111sddd'


def test_validfilename_posixchars():
    """'validfilename' function: 'posixchars' argument"""
    name = ' - --  a*sf/J|<I,;çZ?E->p_%ésv\\64"8d&+:f.gfg. ... .'
    assert validfilename(name, posixchars=True) == 'asfJIZE-p_sv648df.gfg'


def test_validfilename_posixlenght():
    """'validfilename' function: 'posixlenght' argument"""
    name = '12345678901234567890'
    # Truncate case
    assert validfilename(name, posixlenght=True) == '12345678901234'
    # Error case
    with pytest.raises(ValueError) as excinfo:
        validfilename(name, posixlenght=True, lenghterror=True)
    assert 'Filename too long for POSIX' in str(excinfo.value)


def test_validfilename_msdoslenght():
    """'validfilename' function: 'msdoslenght' argument"""
    # Truncate basename and extension
    name = '1234567890.123456'
    assert validfilename(name, msdoslenght=True) == '12345678.123'
    # Error on basename
    name = '1234567890.123'
    with pytest.raises(ValueError) as excinfo:
        validfilename(name, msdoslenght=True, lenghterror=True)
    assert 'Filename too long for MS-DOS' in str(excinfo.value)
    # Error on extension
    name = '12345678.123456'
    with pytest.raises(ValueError) as excinfo:
        validfilename(name, msdoslenght=True, lenghterror=True)
    assert 'Extension too long for MS-DOS' in str(excinfo.value)


def test_validfilename_msdosnames():
    """'validfilename' function: with MS-DOS reserved names"""
    with pytest.raises(ValueError) as excinfo:
        for name in ('CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
                     'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1',
                     'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8',
                     'LPT9'):
            validfilename(name)
    assert 'Filename is a Windows/MS-DOS reserved name' in str(excinfo.value)


def test_validfilename_empty():
    """'validfilename' function: with 'empty' str"""
    name = ' '
    with pytest.raises(ValueError) as excinfo:
        validfilename(name)
    assert 'All characters in filename are invalid' in str(excinfo.value)
