"""Package installation"""

import setuptools

setuptools.setup(
    name='mdi_analysis_pipeline',
    url='https://github.com/jbestenlehner/mdi_analysis_pipeline',
    author='Joachim Bestenlehner',
    author_email='j.m.bestenlehner@sheffield.ac.uk',
    install_requires=['numpy', 'scipy', 'astropy', 'time', 'pandas', 'multiprocessing', 'ctypes'],
    packages=setuptools.find_packages(),
    version='0.0.1',
    license='',
    description='Spectroscopic analysis pipeline using model de-idealisation',
    long_description=open('README.md').read(),
)
