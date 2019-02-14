'''"pypi.python.org" Packaging'''
# TODO:
# - Read version.
# - Import Git build version.
# - Tool for manage optional requierements.
# - Easy extensible package.

from setuptools import setup

# Import infrmations from some files
with open('README.RST', encoding='utf-8') as f:
    README = f.read()

with open('skio/version.py', encoding='utf-8') as f:
    for line in f:
        if 'VERSION =' in line:
            VERSION = line.split('=')[1].strip("' ")
            break

# Set Setuptools setup
setup(
    # Package informations
    name='scikit-io',
    version=VERSION,
    description=('Specific and proprietary data file formats I/O SciKit '
                 '(Toolbox for SciPy)'),
    long_description=README,
    url='https://github.com/scikit-io/scikit-io',
    author='Scikit-I/O team',
    license='BSD',

    # Pypi classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Manufacturing',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='data file formats io',

    # Package directory
    packages=['skio'],

    # Mandatory requierements
    install_requires=['numpy', 'pandas'],

    # Optional requierements
    extras_require={
        'hdf5': ['h5py'],
        'dev':  ['pytest', 'pytest-cov'],
    },
)
