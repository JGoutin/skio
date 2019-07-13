"""System and file management utilities"""

import os.path
import re


def validfilename(filename, fullpath=False, posixchars=False, iso9660=False,
                  posixlenght=False, msdoslenght=False, lenghterror=False):
    r"""
    Remove all invalid characters from a file or folder name and check its
    validity on Linux, Microsoft Windows, Microsoft MS-DOS and Apple Macintosh.

    Remove:

    - All characters <= 31 on ASCII table (Linux, Windows, Macintosh).\n
    - Following special characters: "\", "/", ":", "*", "?", '"', ">", "<" and
      "|" (Windows).
    - " " on start and end of names.
    - "." on end of names (Windows).
    - "-" on start of names (Linux).

    Check also for Windows/MS-DOS reserved names:
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4","COM5", "COM6",
    "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6",
    "LPT7", "LPT8", "LPT9".

    Parameters
    ----------
    filename : str
        File or folder name or path (see "fullpath" parameter).
    fullpath : bool, optional
        Set to "True" if "filename" contain full path. Set to "False"
        if "filename" contain only file or folder name to check.
    posixchars : bool, optional
        If "True", remove all unauthorized characters with POSIX specification.
        With this, only alphanumeric, ".", "-" and "_" are authorized.
    iso9660 : bool, optional
        If "True", remove all "-" that are incompatible with ISO9660 level 1
        optic disk formatting.
    posixlenght : bool, optional
        If "True", check if length is greater than 14.
    msdoslenght : bool, optional
        If "True", check if length is greater than 8 for name and 3 for
        extension.
    lenghterror : bool, optional
        If "True", raise error if length is invalid, else, truncate filename.

    Return
    -------
    out : str
        Fixed filename.
    """

    # Split directory and name
    if fullpath:
        directory, filename = os.path.split(filename)
    else:
        directory = ""

    # Remove invalid characters
    if posixchars:
        # Remove POSIX invalid characters
        validname = re.sub("[^a-zA-Z0-9_.-]", "", filename)
    else:
        # Remove Windows and ASCII<31 invalid characters
        validname = ""
        for char in filename:
            if not (char in '\/:*?"><|') and ord(char) > 31:
                validname += char

    if iso9660:
        # Remove '-' for ISO9660
        validname = re.sub("[-]", "", validname)

    # Remove ending and starting characters that can generate OS errors
    def checkendstart(string):
        """- ' ', '.' on end, '-' on start"""
        prevlen = 0
        while len(string) != prevlen:
            prevlen = len(string)

            # Remove spaces on start and end
            string = string.strip()

            # Remove '.' on end
            string = string.rstrip('.')

            # Remove '-' on start
            string = string.lstrip('-')

        return string

    validname = checkendstart(validname)

    # Check if filename is not empty
    if not validname:
        raise ValueError('All characters in filename are invalid')

    # Check MS-DOS length
    if msdoslenght:
        base, ext = os.path.splitext(validname)
        if len(base) > 8:
            if lenghterror:
                raise ValueError('Filename too long for MS-DOS (8 characters)')
            else:
                # Truncate basename
                validname = base[:8]
        if len(ext) > 4:
            if lenghterror:
                raise ValueError('Extension too long for MS-DOS '
                                 '(3 characters)')
            else:
                # Truncate extension
                validname += ext[:4]
        validname = checkendstart(validname)

    # Check POSIX length
    if posixlenght and len(validname) > 14:
        if lenghterror:
            # Raise error
            raise ValueError('Filename too long for POSIX (14 characters)')
        else:
            # Truncate name
            validname = checkendstart(validname[:14])

    # Check Windows/MS-DOS reserved name:
    if validname in ('CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3',
                     'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1',
                     'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8',
                     'LPT9'):
        raise ValueError("Filename is a Windows/MS-DOS reserved name")

    # Return valid filename
    if directory:
        validname = os.path.join(directory, validname)
    return validname
