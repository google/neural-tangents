# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Setup the package with pip."""


import os
import setuptools

# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'jaxlib>=0.1.37',
    'jax>=0.1.57',
    'frozendict'
]


setuptools.setup(
    name='neural-tangents',
    version='0.1.6',
    license='Apache 2.0',
    author='Google',
    author_email='neural-tangents-dev@google.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/google/neural-tangents',
    download_url = "https://pypi.org/project/neural-tangents/",
    project_urls={
        "Source Code": "https://github.com/google/neural-tangents",
        "Documentation": "https://arxiv.org/abs/1912.02803",
        "Bug Tracker": "https://github.com/google/neural-tangents/issues",
    },
    packages=setuptools.find_packages(exclude=('presentation',)),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Fast and Easy Infinite Neural Networks in Python',
    python_requires='>=3.5',
    classifiers=[
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
    ]
)
