"""'Group' dict-like class definition"""
# TODO:
# - Find how show parameters and doc-strings for .set()/.get().
# - Orderable Group.
# - Autocomplete __doc__ with _doc, _dtype, ... informations for subclasses.
# - Add unit information support with optional support of "pint" package.
#   store data in group values without unit (and convert on get or set)
# - Support for data sample/uncertainty.
# - Add _dtype arguments for numpy: shape (Fixed, min, max)
# - Add _dtype arguments for pandas: nbcols, nbrows (Fixed, min, max)
# - Custom _dtype args
# - Write public name ("Group.get(key, ...)") in errors on advanced functions
#   produced by _customfunc in place of private '_get_key' like name. Also
#   include arguments if possible.
# - .clear(): not remove inserted classes, but reset them
# - in "set", make type support extensible and optionnal, starting with numpy

from collections import Mapping
from collections.abc import Iterable
from contextlib import contextmanager
from sys import getsizeof
from inspect import isclass
from copy import copy, deepcopy
from numpy import ndarray, array, ma
from itertools import chain
try:
    import pint
    UREG = pint.UnitRegistry()
except ImportError:
    UREG = None
    from warnings import warn
    warn('Pint package required for advanced Units support')


class Group(Mapping):
    """
    Advanced dict with optionnal features (See bellow).

    The aim of this class is to have a dict like data structure with more
    controls of data inside.

    This classe is intended to be subclassed and not to be used directly.

    Subclassing for activating features
    -----------------------------------
    Subclassing this class, overloading some class variables and create methods
    with specific names is needed for activating following features:

    Set default values: overloading the "_default" class variable
        If Key not find in subclass instance, default value is automatically
        returned.

        Overload the "_default" class variable with a dict containing the
        needed default value for each required key.

        Exemple: _default = {'key1': 1.2, 'key2': 3}

    Type values: overloading the "_dtype" class variable
        When value is set, value is automatically casted to specified type or
        error is raised. A "None" value is set as "None" directly for and may
        be used as no data.

        Overload the "_dtype" class variable with a dict containing the
        needed type for each required key.

        Exemple: _dtype = {'key1': float, 'key2': np.int16}

        This feature support also more advanced typing. It is possible to add
        named arguments for the specified type setting it with a tuple
        like : (type, {argname1: argvalue1, argname2: argvalue2})

        For numpy arrays, "ndim" int argument is added and return error if
        input array number of dimensions is not equal to "ndim".

        Exemple: _dtype = {'key1': (numpy.ndarray, {'ndim': 2, 'dtype':
        np.float64})}

    Documented keys: overloading the "_doc" class variable
        Return doc_string for a specified key with instance .doc(key) method.
        Keys docstrings are also appened to the end of the subclass docstring.

        Overload the "_doc" class variable with a dict containing the
        needed docstrings for each required key.

        Exemple: _dtype = {'key1': "A float number", 'key2': "A integer
        number"}

    Advanced Getter/Setter functions: Create funtions with specific names
        Theses functions replace standards way to get and set the value linked
        to a key in subclass instances with possible more powerfull functions
        and the ability to use them with more than one argument.

        Getter functions are called with the ".get(key, *args, **kwargs)"
        method. The function must have possibility to be called with only one
        argument since value can also be got with usual dict way
        "value = MyInstance[key]". The function can have any number of
        optionals arguments.
        The function must return the value to get.

        If needed, you can access to the raw value (or default value if
        missing) as stored in the subclass directly with "_get_raw()" method.

        Setter functions are called with the ".set(key, *args, **kwargs)"
        method. If the function must have possibility to be called with only
        one argument since value can also be set with usual dict way
        "MyInstance[key] = value". The function can have any number of
        optionals arguments.
        The function must return the value to set.

        For create a getter/setter function, simply create a method or a
        static method called "_get_name" (For the getter) and/or a method
        called "_set_name" (For the setter) with "name" replaced by the key.
        If key is not a string or is a non-python compatible name for a
        function, you can use the "_funcbase" class variable to link key and
        function (See below).

        It is also possible to overload the "_funcbase" class variable with a
        dict containing the alias function basename for each required key.
        The alias is used to name the getter/setter functions linked to the
        key. If "_funcbase" is not overloaded for a key, try to call function
        using directly the key.

        Exemple: _funcbase = {'key1,12': "k112"}; for the key 'key1,12' key,
        setter will be "_set_k112" and getter "_get_k112".

        "_funcbase" can also be used to redirect many key getter/setter on a
        same function.

        Exemple: _funcbase = {'key1': "root", 'key2': "root"}

    Inserting other Groups or classes as Key and auto-instanciate them:
        For adding groups, simply write your Group subclass definition inside
        the File Subclass directly. The class method must be "_Keyname" with
        "name" the key name to find in File.keys().

        Once File is instancied, all classes inside it starting with a
        "_Keyname" name will be automatically instanciated and will be
        availables in File dictionnary interface.

        Exemple: for "_keyname" class definied inside File, instance access
        with "FileInstance['name']". File instance is available from Group
        instance with the "prt" or "parent" attribute.

        The "_Keyname" mechanic also work with classes that are not Group.

        The added class is added to Default values and not on registered
        values. So it not count for len(), iter(), ...

    Writing limitation: overloading "_readonly" or "_nonewkey" class variable
        If "_readonly" is True, the subclass will be read only.

        If "_nonewkey" is True, creation of new keys is forbiden.
        Values for keys that already have default values can still be modified.

        All of theses values are False by default.

        You can use the "_writeenabled" context manager to temporabilly
        re-enable writing.

    Overloading "__init__()" method:
        The default "__init__()" only write values from "mapping" parameter
        inside instance.

        So, "__init__()" can safely be overloaded without calling
        "Group.__init__(self, mapping)" in subclasses.

    Parameters
    ----------
    Parameters with the default "__init__()" method of Group.

    mapping : dict compatible, optional
        Used to populate instance with data. See dict documentation.
        Empty by default.
    """
    # Overloadable class variables
    _default = {}  # Default values dict (Used if no registered value)
    _dtype = {}  # Values types
    _funcbase = {}  # Setter/getter functions basenames
    _doc = {}  # Help/doc strings
    _readonly = False  # Read only flag
    _nonewkey = False  # New key limitation flag

    def __new__(cls, *args, **kwargs):
        """
        Group Instanciation.
        """
        # Create instance
        self = Mapping.__new__(cls)

        # Set default Parent value: in case not called by another Group
        self._parent = None
        self._values = {}
        self._iterdefault = False
        self._name = cls.__name__

        # Performance: Alias dotted names
        clsdict = cls.__dict__
        clsdtype = cls._dtype
        clsdoc = cls._doc
        values = self._values

        # Instanciate attributes-classes
        module = ("{0}['{1}']".format(cls.__module__, cls.__name__))
        for key in clsdict.keys():
            attr = clsdict[key]

            # Only for classes starting with '_Key'
            if not isclass(attr):
                continue
            name = attr.__name__
            if not name.startswith('_Key'):
                continue

            # Update attribute-class informtations
            name = name[len('_Key'):]
            attr.__name__ = name
            attr.__module__ = module

            # Instanciate attribute-class
            instance = attr()

            # If attribute-class is Group, add informations
            if isinstance(instance, Group):
                setattr(instance, '_parent', self)
                if name not in clsdtype:
                    clsdtype[name] = Group

            # Add doc to current class _doc
            if name not in clsdoc:
                clsdoc[name] = attr.__doc__

            # Add instance in default value dict
            values[name] = instance

        # Return instance
        return self

    def __init__(self, mapping=None):
        if mapping:
            self._values.update(mapping)

    @property
    def parent(self):
        """
        Parent of this instance. Can be any object.
        """
        return self._parent

    prt = parent  # Alias for "parent" property

    def _customfunc(self, key, action, *args, **kwargs):
        """
        Return result of an advanced function for the specified key.
        Key and function basename should be registered in "_funcbase" class
        variable (if not, key is used directly to search function name).

        Parameters
        ----------
        key : object
            key.
        action : str
            Action to do. Can be "get" or "set".
        *args, **kwargs :
            Arguments to pass to the function.
        """
        name = '_{0}_{1}'.format(action, self._funcbase.get(key, key))
        func = getattr(self, name, None)

        # No function to return
        if func is None:
            raise _NoFunctionFoundError

        # Return function
        return func(*args, **kwargs)

    def set(self, key, *args, **kwargs):
        """
        Set the value for key.

        Parameters
        ----------
        key : object
            key.
        *args, **kwargs : optionnal
            Value and/or other arguments (Specific to each key).
        """
        if isinstance(self._values.get(key), Group):
            raise PermissionError('Groups are not overwritable')
        if self._readonly:
            raise PermissionError('{} is read only'.format(self._name))
        if self._nonewkey and key not in self.keys_all():
            raise PermissionError('New key creation is forbidden')

        try:
            # Use function with full set of args
            newvalue = self._customfunc(key, 'set', *args, **kwargs)
        except _NoFunctionFoundError:
            # Try other ways
            newvalue = args[0]

        # Get data type
        dtype = self._dtype.get(key, object)
        if isinstance(dtype, Iterable):
            # Advanced typing
            kwargs = dtype[1]
            dtype = dtype[0]
        else:
            # Classic typing
            kwargs = {}

        if newvalue is None:
            # no data
            pass

        elif dtype is object:
            # Everything is object
            pass

        elif issubclass(dtype, ndarray):
            # Special case of ndarray
            npkwargs = deepcopy(kwargs)

            if 'copy' not in npkwargs:
                # Reference by default
                npkwargs['copy'] = False

            if 'ndim' in npkwargs:
                # For number of dimensions check
                ndim = npkwargs['ndim']
                del npkwargs['ndim']
            else:
                ndim = 0

            if dtype is ndarray:
                newvalue = array(newvalue, **npkwargs)
            else:
                newvalue = dtype(newvalue, **npkwargs)

            if ndim > newvalue.ndim:
                # Check number of dimensions
                raise ValueError('Array of {} dimensions needed'.format(ndim))
        elif kwargs:
            # Set type with advanced typing
            newvalue = dtype(newvalue, **kwargs)

        elif not isinstance(newvalue, dtype):
            # Check type and cast if not correct type
            newvalue = dtype(newvalue)

        self._values[key] = newvalue

    __setitem__ = set

    def get(self, key, *args, **kwargs):
        """
        Return the value for key.

        Parameters
        ----------
        key : object
            key.
        *args, **kwargs : optionnal
            Other arguments (Specific to each key).
        """
        try:
            # Use function with full set of args
            return self._customfunc(key, 'get', *args, **kwargs)
        except _NoFunctionFoundError:
            # Try other ways
            return self._get_raw(key)

    __getitem__ = get

    def _get_raw(self, key):
        """
        Return the raw value for a key directly (or default raw value if
        missing).

        Parameters
        ----------
        key : object
            key.
        """
        try:
            return self._values[key]
        except KeyError:
            try:
                return self._default[key]
            except KeyError:
                return self.__missing__(key)

    def __missing__(self, key):
        """
        Raise error if key not regidtered and no default value.

        Parameters
        ----------
        key : object
            key.
        """
        raise KeyError('No registered or default value for {0!r}'.format(key))

    def __contains__(self, key):
        """
        Return True if key is registered.

        Parameters
        ----------
        key : object
            key.
        """
        return key in self._values

    def __delitem__(self, key):
        """
        Delete the registered value for key.

        Parameters
        ----------
        key : object
            key.
        """
        if isinstance(self._values.get(key), Group):
            raise PermissionError('Groups are not overwritable')
        if self._readonly:
            raise PermissionError('{} is read only'.format(self._name))

        del self._values[key]

    def clear(self):
        """
        Remove all items.
        """
        if self._readonly:
            raise PermissionError('{} is read only'.format(self._name))

        self._values.clear()

    def copy(self, deep=True):
        """
        Return a copy of this object.

        Parameters
        ----------
        deep : bool, optional
            If True, create a deep copy.
            If False, create a shallow copy.

        Return
        ------
        Same type as this object.
        """
        return deepcopy(self) if deep else copy(self)

    def update(self, source, deep=False):
        """
        Update this object with the key/value pairs from source,
        overwriting existing keys.

        Parameters
        ----------
        source : dict like
            Source for update.
        deepcopy : bool, optional
            If True, updade with deep copies of source pairs.
            If False, updade with references of source pairs.
        """
        if self._readonly:
            raise PermissionError('{} is read only'.format(self._name))

        for key, value in (deepcopy(source) if deep else source).items():
            try:
                self.set(key, value)
            except PermissionError:
                continue

    def keys(self):
        """
        Return a list of registered keys.
        """
        with self._iterall(False):
            return list(self)

    def keys_all(self):
        """
        Return a list of registered keys and keys with default values.
        """
        with self._iterall(True):
            return list(self)

    def values(self):
        """
        Return a list of registered values.
        """
        with self._iterall(False):
            getvalue = self.get
            return [getvalue(key) for key in self]

    def values_all(self):
        """
        Return a list of registered and default values.
        """
        with self._iterall(True):
            getvalue = self.get
            return [getvalue(key) for key in self]

    def values_raw(self, default=False):
        """
        Return a list of registered values.

        Parameters
        ----------
        default : bool
            If True, include default values.
        """
        # Raw registered + default values list
        if default:
            with self._iterall():
                getvalue = self._get_raw
                return [getvalue(key) for key in self]
        # Raw registered values list
        else:
            return list(self._values.values())

    def items(self):
        """
        Return a list of registered items.
        """
        with self._iterall(False):
            getvalue = self.get
            return [(key, getvalue(key)) for key in self]

    def items_all(self):
        """
        Return a list of registered and default items.
        """
        with self._iterall(True):
            getvalue = self.get
            return [(key, getvalue(key)) for key in self]

    def items_raw(self, default=False):
        """
        Return a list of registered items.

        Parameters
        ----------
        default : bool
            If True, include unregistered items with default values.
        """
        # Raw registered + default items list
        if default:
            with self._iterall():
                getvalue = self._get_raw
                return [(key, getvalue(key)) for key in self]

        # Raw registered items list
        else:
            return list(self._values.items())

    def default(self, key):
        """
        Return the default value for a key.

        Parameters
        ----------
        key : object
            key.
        """
        try:
            return self._default[key]
        except KeyError:
            raise KeyError(r'No default value found for {0!r}'.format(key))

    def dtype(self, key):
        """
        Delete the required type for the value stored for key.

        Parameters
        ----------
        key : object
            key.
        """
        dtype = self._dtype.get(key, object)
        if isinstance(dtype, Iterable):
            return dtype[0]
        return dtype

    def doc(self, key):
        """
        Show the documentation for the specified key.

        Parameters
        ----------
        key : object
            key.
        """
        return self._doc.get(key, '')

    def asdict(self, default=False, deep=False):
        """
        Return a dictionnary.

        Parameters
        ----------
        default : bool
            If True, include unregistered items with default values.
        deep : bool, optional
            If True, create a deep copy.
            If False, create a shallow copy.
        """
        dictionnary = {}
        with self._iterall(default):
            getvalue = self.get
            for key in self:
                value = getvalue(key)
                if isinstance(value, Group):
                    # Convert to dict Group inside this one
                    value = value.asdict(default)
                if deep:
                    value = deepcopy(value)
                    key = deepcopy(key)
                dictionnary[key] = value
        return dictionnary

    def __repr__(self):
        """
        Class representation : Group(Key <Type>: Value <Default>, ...)
        """
        reprlist = [self._name,
                    '(\nKey <Value type>: Value preview <Default value flag>']

        # Performance: Alias dotted names
        getvalue = self.get
        gettype = self._dtype.get
        keys = self.keys()

        for key in self.keys_all():
            # Get value
            valuerepr = repr(getvalue(key))

            # Reduce value repr to one line of max 61 characters
            if valuerepr.find('\n') > -1:
                valuelist = valuerepr.split('\n')
                valuerepr = '[...]'.join((valuelist[0][:27].strip(),
                                          valuelist[-1][:27].strip()))
            elif len(valuerepr) > 61:
                valuerepr = '[...]'.join((valuerepr[:27].strip(),
                                          valuerepr[-27:].strip()))

            # Get type info
            dtype = gettype(key, object)
            if isinstance(dtype, Iterable):
                dtype, kwargs = dtype
                typename = dtype.__name__
                # Add more informations for ndarrays
                if issubclass(dtype, ndarray):
                    typelst = [typename]
                    if 'dtype' in kwargs:
                        typelst.append("dtype: {[dtype].__name__}".
                                       format(kwargs))
                    if 'ndim' in kwargs:
                        typelst.append("dim: {[ndim]}".format(kwargs))

                    typename = ', '.join(typelst)
            else:
                typename = dtype.__name__

            # Show if value is default
            if not issubclass(dtype, Group) and key not in keys:
                default = ' <default>'
            else:
                default = ''

            # Write line
            reprlist.extend(['\n', repr(key), ' <', typename, '>: ', valuerepr,
                             default])
        reprlist.append(')')
        return ''.join(reprlist)

    def __iter__(self):
        """
        Iterate over registered keys.
        """
        if self._iterdefault:
            # Return keys from registered values and default values
            return iter(set(chain(self._values, self._default)))
        else:
            # Return keys from registered values only
            return iter(self._values)

    def __len__(self):
        """
        Return len of registered values.
        """
        return len(self._values)

    def __sizeof__(self):
        """
        Return size in bytes of this object.
        """
        # Initialize temporary variables
        seen = set()
        size = 0

        # Performance: Alias dotted names
        values = self._values
        cls_maskedarray = ma.core.MaskedArray
        see = seen.add

        for key in self:
            value = values[key]
            dtype = type(value)

            # Case of numpy based arrays
            if issubclass(dtype, ndarray):
                # List all arrays in object
                if issubclass(dtype, cls_maskedarray):
                    # If Numpy masked array
                    items = [value.data, value.mask]
                else:
                    # Classic ndarray
                    items = [value]

                # Return size for each sub-array
                for item in items:
                    base = item.base
                    itemid = id(item)
                    baseid = id(item)
                    nbytes = item.nbytes
                    if ((base is None or baseid not in seen) and
                            itemid not in seen and nbytes):
                        see(itemid)
                        if base is not None:
                            see(baseid)
                        size += nbytes

            # Case of other Python objects
            else:
                valueid = id(value)
                nbytes = getsizeof(value)
                if valueid not in seen and nbytes:
                    see(valueid)
                    size += nbytes

        return size

    @contextmanager
    def _writeenabled(self, nonewkey=False):
        """
        This context manager temporary enable writing if _readonly and/or
        _nonewkey is set to True.

        Parameters
        ----------
        nonewkey : bool
            If True, _nonewkey is not temporally set to False.
        """
        # Back up values
        readonly = self._readonly
        nonewkey = self._nonewkey

        # Enable write
        self._readonly = False
        self._nonewkey = nonewkey

        yield

        # Restore values
        self._readonly = readonly
        self._nonewkey = nonewkey

    @contextmanager
    def _iterall(self, default=True):
        """
        This context manager make iteration over Group also yield default
        values.

        Parameters
        ----------
        default : bool
            If True, iterate also default values.
        """
        self._iterdefault = default
        yield
        self._iterdefault = False


class _NoFunctionFoundError(Exception):
    """Raised when no function found"""
