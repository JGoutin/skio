# -*- coding: utf-8 -*-
"""Scikit-io"""
# TODO :
# - docstring for this module.
# - 'XYZ' like type (With pandas, Any nb of dim, X & Y as 'coordinates',
#    Z as data) and conversion to 'grid' (ndarray of Z) like
# - Viewer UI
# - Make all dependencies optionals, starting by numpy and pandas
# - Lazzy import a maximum of functions

from skio.version import VERSION as __version__

# Populate namespace with useful names only
from skio.file import File, FileFormatError
from skio.group import Group
from skio.system import validfilename
from skio.codec import intdecode, intencode
import skio.formats as formats

for _func in (Group, File, FileFormatError, intencode, intdecode,
              validfilename):
    _func.__module__ = _func.__module__[:_func.__module__.rfind('.')]
del _func
