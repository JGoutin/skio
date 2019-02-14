"""Tests for skio/group.py"""
from skio import Group
import pytest
import numpy as np
import sys


# Test class and instance
class Example01(Group):
    """Test Group"""
    _default = {'01': 1, '02': 2.1, '03': None}
    _dtype = {'01': int, '03': (np.ndarray, {'ndim': 2, 'dtype': np.float64}),
              '04': (np.ma.MaskedArray, {'copy': True}),
              '35': (str, {'encoding': 'utf-16'})}
    _funcbase = {'02': 'f02'}
    _doc = {'01': 'doc 1', '02': 'doc 2'}

    @staticmethod
    def _set_f02(value, square=False):
        """Test setter"""
        result = value * 10
        return result**2 if square else result

    def _get_f02(self, square=False):
        """Test getter"""
        return ((self._get_raw('02') * 10)**2 if square else
                self._get_raw('02') * 10)

    def _set_f12(self, value, default=False):
        """Test setter"""
        return self.default('02') if default else value

    @staticmethod
    def _get_f12(square=False):
        """Test getter"""
        return 100 if square else 10


class Example02(Group):
    """Test Group"""
    class _Keygroup(Group):
        """Test inserting other group"""
        pass

    class _Group(Group):
        """Test inserting other group with bad name"""
        pass

EXAMPLE01 = Example01({'10': 1.5})
EXAMPLE02 = Example02()


def test_group_default():
    """'Group' class: Default values"""
    items = {'01': 1, '03': None}

    for key in items:
        # Default value if not set
        assert EXAMPLE01[key] == items[key]
        # "default" property
        assert EXAMPLE01.default(key) == items[key]

    # Not default after value change
    EXAMPLE01['01'] = 3
    assert EXAMPLE01['01'] != items['01']

    # Default after value reset
    del EXAMPLE01['01']
    assert EXAMPLE01['01'] == items['01']

    # Error if no default
    with pytest.raises(KeyError) as excinfo:
        EXAMPLE01.default('10')
    assert 'No default value found for' in str(excinfo.value)

    with pytest.raises(KeyError) as excinfo:
        assert EXAMPLE01['20']
    assert 'No registered or default value for' in str(excinfo.value)


def test_group_dtype():
    """'Group' class: Types"""
    # "dtype" property
    items = {'01': int, '03': np.ndarray}
    for key in items:
        assert EXAMPLE01.dtype(key) is items[key]


def test_group_set_get():
    """'Group' class: setter"""
    # Setting false type, casted
    EXAMPLE01['01'] = 2.0
    assert isinstance(EXAMPLE01['01'], int)

    # Setting false type, incompatible
    with pytest.raises(TypeError) as excinfo:
        EXAMPLE01['01'] = str
    assert "not 'type'" in str(excinfo.value)

    # Change type if no dtype set
    EXAMPLE01['02'] = 2
    assert isinstance(EXAMPLE01['02'], int)

    # Setting None value
    EXAMPLE01['03'] = None
    assert EXAMPLE01['03'] is None

    # setting NumPy array with wrong dimensions
    with pytest.raises(ValueError) as excinfo:
        EXAMPLE01['03'] = np.ones(5)
    assert 'Array of 2 dimensions needed' in str(excinfo.value)

    # Setting NumPy array with wrong dtype
    EXAMPLE01['03'] = np.ones((2, 2), dtype=np.int16)
    assert EXAMPLE01['03'].dtype is np.dtype('float64')

    # Setting NumPy masked array
    data = np.ma.array(((1, 1), (1, 1)), mask=((1, 0), (0, 1)))
    EXAMPLE01['03'] = data
    assert not isinstance(EXAMPLE01['03'], np.ma.MaskedArray)
    EXAMPLE01['04'] = data
    assert isinstance(EXAMPLE01['04'], np.ma.MaskedArray)

    # Type with more than one arg
    EXAMPLE01['35'] = b'\xff\xfe1\x002\x003\x004\x00'
    assert EXAMPLE01['35'] == '1234'

    # Setter with more than 1 args
    EXAMPLE01.set('f12', 1, False)

    # Getter with more than 1 args
    assert EXAMPLE01.get('f12', False) == 10

    # .get with incorrect arguments
    with pytest.raises(TypeError):
        EXAMPLE01.get('f12', 1, 1)

    # .set with incorrect arguments
    with pytest.raises(TypeError):
        EXAMPLE01.set('f12', 1, 1, 1)

    # .set a group
    with pytest.raises(PermissionError):
        EXAMPLE02.set('group', 1)

    # New key
    example03 = Example01()
    example03['20'] = 2
    assert '20' in example03

    # No New key
    example03._nonewkey = True
    example03['01'] = 2
    assert example03['01'] == 2
    with pytest.raises(PermissionError):
        example03['30'] = 2

    # No Change
    example03._nonewkey = False
    example03._readonly = True
    with pytest.raises(PermissionError):
        example03['01'] = 2
    with pytest.raises(PermissionError):
        example03['30'] = 2
    with pytest.raises(PermissionError):
        del example03['01']

    # Temporary write context manager
    with example03._writeenabled():
        example03['01'] = 2
        assert example03['01'] == 2


def test_group_insert_other_groups():
    """'Group' class: inserting Group"""
    assert isinstance(EXAMPLE02['group'], EXAMPLE02._Keygroup)
    assert EXAMPLE02['group'].prt is EXAMPLE02
    assert EXAMPLE02['group'].parent is EXAMPLE02

    # no parent if not inserted group
    assert EXAMPLE02.prt is None

    # Class with bad name (no "Key" on start)
    assert 'Group' not in EXAMPLE02

    # Not overwrite group
    with pytest.raises(PermissionError):
        del EXAMPLE02['group']


def test_group_dict_methods():
    """'Group' class: generic dict methods"""
    example03 = Example01({'01': 2, '10': 4})

    def checklist(lista, listb):
        """Check if all elements of lista in listb"""
        if len(lista) != len(listb):
            return False
        for value in lista:
            if value not in listb:
                return False
        return True

    # .keys()
    assert checklist(example03.keys(), ('01', '10'))
    assert checklist(EXAMPLE02.keys(), ('group',))

    # .keys_all()
    assert checklist(example03.keys_all(), ('01', '02', '03', '10'))

    # .values()
    assert checklist(example03.values(), (2, 4))

    # .values_all()
    assert checklist(example03.values_all(), (None, 2, 21.0, 4))

    # .values_raw()
    assert checklist(example03.values_raw(True), (None, 2, 2.1, 4))
    assert checklist(example03.values_raw(False), (2, 4))

    # .items()
    assert checklist(example03.items(), (('01', 2), ('10', 4)))

    # .items_all()
    assert checklist(example03.items_all(),
                     (('01', 2), ('02', 21.0), ('03', None), ('10', 4)))

    # .items_raw()
    assert checklist(example03.items_raw(True),
                     (('01', 2), ('02', 2.1), ('03', None), ('10', 4)))
    assert checklist(example03.items_raw(False), (('01', 2), ('10', 4)))

    # .__contains__()
    assert '01' in example03
    assert '20' not in example03

    # .__len__()
    assert len(example03) == 2

    # .update()
    example03 = Example01({'01': 2, '10': 4})
    example03.update({'11': 0, '01': 5}, False)
    assert '11' in example03
    assert example03['01'] == 5
    example03.update({'11': 0, '01': 5}, True)
    assert '11' in example03
    assert example03['01'] == 5

    # .update(), but read only
    example03 = Example01({'01': 2, '10': 4})
    example03._readonly = True
    with pytest.raises(PermissionError):
        example03.update({'11': 0, '01': 5}, False)
    assert '11' not in example03

    # .update(), but no new keys
    example03 = Example01({'01': 2, '10': 4})
    example03._nonewkey = True
    example03.update({'11': 0, '01': 5}, False)
    assert '11' not in example03
    assert example03['01'] == 5

    # .clear()
    example03 = Example01({'01': 2, '10': 4})
    example03.clear()
    assert len(example03) == 0

    # .clear(), but read only
    example03 = Example01({'01': 2, '10': 4})
    example03._readonly = True
    with pytest.raises(PermissionError):
        example03.clear()
    assert len(example03) == 2

    # .copy()
    assert example03 == example03.copy(True)
    assert example03 == example03.copy(False)


def test_group_asdict():
    """'Group' class: asdict"""
    example03 = Example01({'01': 2, '10': 4})
    assert example03.asdict(False, True) == {'01': 2, '10': 4}
    assert isinstance(EXAMPLE02.asdict(False, False)['group'], dict)


def test_group_doc():
    """'Group' class: keys documentation"""
    doc = {'01': 'doc 1', '02': 'doc 2'}
    for key in doc:
        assert EXAMPLE01.doc(key) == doc[key]


def test_group_repr():
    """'Group' class: return a repr"""
    example03 = Example01({'01': 2, '10': 4, '23': 'z' * 80,
                          '24': np.ones((2, 2))})
    assert repr(example03) != ''


def test_group_sizeof():
    """'Group' class: return a size"""
    assert sys.getsizeof(EXAMPLE01) > 0
