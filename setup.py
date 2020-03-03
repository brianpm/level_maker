import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="level_maker",
    version="0.2.0",
    url="https://github.com/brianpm/level_maker",
    license='MIT',

    author="Brian Medeiros",
    author_email="brianpm@ucar.edu",

    description="NCAR/CGD/AMP Generator for atmosphere hybrid-sigma coefficients.",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
