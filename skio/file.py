# -*- coding: utf-8 -*-
"""'File' definition"""
# TODO:
# - HDF5 Export and import (For files exported with skio).
# - Average files
# - File customizable UnitRegistry
# - Make HDF5 support and other as optionals extensions

from skio.group import Group
from datetime import datetime
from os.path import getsize, getctime, getmtime
from inspect import ismethod
try:
    import h5py
    HDF5ERROR = ''
except ImportError:
    HDF5ERROR = 'h5py package required for HDF5 Import/Export'
    from warnings import warn
    warn(HDF5ERROR)


class File(Group):
    """
    Create a new skio.File.

    The skio.File class work like a special Python dictionary.
    Datas that are stored inside will be called with the following rule:

    Group :
        Sub-dictionary (skio.Group subclass) which can contain any dataset,
        attribute or group.
    Dataset :
        Data (numpy.ndarray, pandas.DataFrame, ...)
    Attribute :
        Metadata (any other data type like int, float, str, list, ...)

    This classe is intended to be subclassed and not to be used directly.

    Subclassing for activating features
    -----------------------------------
    This class is a subclass of Group, so all Group features are also in File.

    Subclassing this class, overloading some class variables and create methods
    and class with specific names is needed for activating following features:

    Store file related information:
        File contain a "info" Group for store some file related information
        (modification date, size, ...).

        The list of supported information is read only but can be
        extended by overloading "_infos_default", "_infos_dtype" and
        "_infos_doc" File class variables excatly like in "Group" class
        ("_infos_default" work like "_default", "_infos_dtype" like
        "_dtype" and "_infos_doc" like "_doc").

        For update values of "info" Group, use the "_updateinfos" method. Base
        information  (Like modification date, ...) are automatically updated
        based on the "filename", extra information (With support added by
        overloading "_infos_default") can also be update with the "extrainfos"
        parameter.

    File loading/saving methods:
        Loading methods names must be "loadext" with "ext" the file extension
        (or another format variation specifier).

        The "load" method is a special method that try to load the file using
        all "loadext" available methods. If there is only one possible
        variation of the format, it is possible to overload it directly.

        Saving methods names must be "saveext" with "ext" the same name
        specifier than the related "loadext" method.

        Like for load, "save" can be used directly if there is only one
        variation of the format.

        Don't forget to call "_updateinfos" in load methods to update the
        "infos" Group.

        If a "load" is used with a file of the bad format, skio.FileFormatError
        exception must be raised.

    Overloading "__init__()" method:
        The default "__init__()" only try to load the file using the "load"
        method.

        So, "__init__()" can safely overloaded without calling
        "File.__init__(self, filename)" in subclasses.

    Default content
    ---------------
    infos : Group
        Contain some file related information (modification date, size, ...)
    subfiles : List of "File"
        A list of sub-"File" related to this File (Exemple: case of multiple
        Data files stored in one physical file)

    Parameters
    ----------
    Parameters with the default "__init__()" method of File.

    filename : str, optional
        If specified: open this file;
    """

    # Extra format specific related keys specifications for 'infos' Group
    _infos_default = {}
    _infos_dtype = {}
    _infos_doc = {}

    # No new key by default
    _nonewkey = True

    def __init__(self, filename=''):
        # Try opening file
        if filename:
            self.load(filename)

    def _updateinfos(self, filename='', extrainfos=None):
        """
        Update all information in the "infos" group.

        Parameters
        ----------
        filename : str, optional
            If specified: Update 'filename' key with specified value.
        """
        self['infos'].updateinfos(filename, extrainfos)

    def load(self, filename):
        """
        Load file (Try to autodetect if many format variations are availables).

        Parameters
        ----------
        filename : str
            file to open.
        """
        tdict = type(self).__dict__
        for key in tdict.keys():
            attr = tdict[key]

            # Only for loading methods
            if not ismethod(attr):
                continue
            name = attr.__name__
            if not name.startswith('load'):
                continue
            if name == 'load':
                continue

            # Try opening file
            try:
                attr(self, filename)
            except FileFormatError:
                continue
            break

    def loadhdf5(self, filename):
        """
        Load from an HDF5 file.

        Parameters
        ----------
        filename : str
            file to open (.h5, .hdf5, .he5).
        """
        if HDF5ERROR:
            raise ImportError(HDF5ERROR)
        raise NotImplementedError

    def savehdf5(self, filename, compression='gzip'):
        """
        Save as HDF5 file.

        For Numpy MaskedArrays, mask are stored in a second dataset named
        'MaskedArrayName_mask' (with 'MaskedArrayName', the name of the
        MaskedArray in skio.File)

        Attributes that are not scalar or Numpy ndarray are converted to
        their Python string representation.

        Parameters
        ----------
        filename : str
            file to save (.h5, .hdf5, .he5).
        compression : str, optional
            Dataset compression filter. Can be 'gzip' (Good compression,
            moderate speed), 'lzf' (Low to moderate compression, very fast) or
            any other filter available with your H5py installation.
        """
        if HDF5ERROR:
            raise ImportError(HDF5ERROR)

        raise NotImplementedError

    class _Keysubfiles(list):
        """
        A list of sub-"File" related to this File (Exemple: case of multiple
        Data files stored in one physical file)
        """

    class _Keyinfos(Group):
        """
        This group contain some file related information. Theses
        information are directly taken from the OS for files on the disk.
        """

        # Set keys specifications
        _dtype = {'filename': str,
                  'datemodification': datetime,
                  'datecreation': datetime,
                  'bytesize': int}
        _doc = {'filename': 'Path to the file',
                'datemodification': 'Date of the last file modification.'
                'Set to current date and time when file not loaded from path.',
                'datecreation': 'Date of the file creation. Set to current '
                'date and time when file not loaded from path.',
                'bytesize': 'Size of the file on disk in bytes'}
        _readonly = True

        def __init__(self, filename=''):
            prt = self.prt

            # Update keys specifications
            for key in ('_default', '_dtype', '_doc'):
                getattr(self, key).update(getattr(prt, '_infos{}'.format(key)))

            # Get file infos
            self._updateinfos(filename)

        def _updateinfos(self, filename='', extrainfos=None):
            """
            Update all file information in this group.

            Parameters
            ----------
            filename : str, optional
                If specified: Update 'filename' key with specified value.
            extrainfos : dict, optional
            """
            with self._writeenabled(nonewkey=True):
                # Update filename
                if filename:
                    self['filename'] = filename
                else:
                    filename = self['filename']

                # Update generic file information
                if not self['filename']:
                    self['datemodification'] = datetime.today()
                    self['datecreation'] = datetime.today()
                    self['bytesize'] = 0
                else:
                    self['datemodification'] =\
                        datetime.fromtimestamp(getmtime(filename))
                    self['datecreation'] =\
                        datetime.fromtimestamp(getctime(filename))
                    self["bytesize"] = getsize(filename)

                # Update extra information
                if extrainfos:
                    self.update(extrainfos)


class FileFormatError(Exception):
    """Raised when a file is not in expected format."""

    msg = "File not in expected format"

    def __init__(self, msg=''):
        if msg:
            self.msg = msg

    def __str__(self):
        return repr(self.msg)
