# Copyright 2015 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
import os
from setuptools import find_packages, setup, Extension

_VERSION = '0.6.2'

REQUIRED_PACKAGES = [
    'enum34 >= 1.0.0',
    'six >= 1.10.0',
    'tensorflow >= 0.9.0rc0',
]

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'prettytensor_model_mnist = prettytensor.tutorial.mnist:main',
    'prettytensor_model_shakespeare = prettytensor.tutorial.shakespeare:main',
    'prettytensor_model_baby_names = prettytensor.tutorial.baby_names:main',
]
# pylint: enable=line-too-long

TEST_PACKAGES = [
    'nose >= 1.3.7',
]

setup(
    name='prettytensor',
    version=_VERSION,
    description='Pretty Tensor makes learning beautiful',
    long_description='',
    url='https://github.com/google/prettytensor',
    author='Eider Moore',
    author_email='opensource@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'prettytensor': ['tutorial/baby_names.csv']
        },
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS
        },
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    test_suite = 'nose.collector',
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning',
    )
