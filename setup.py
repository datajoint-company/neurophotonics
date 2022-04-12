from setuptools import setup, find_packages
from os import path

pkg_name = 'neurophotonics'
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requirements = f.read().splitlines()

with open(path.join('neurophotonics', 'version.py')) as f:
    exec(f.read())

setup(
    name=pkg_name,
    version=__version__,
    description="Neurophotonics probe simulation",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='DataJoint',
    author_email='info@datajoint.com',
    license='MIT',
    url=f'https://github.com/datajoint-company/{pkg_name.replace("_", "-")}',
    keywords='neuroscience photonics science datajoint',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    scripts=[],
    python_requires='>=3.6',
    install_requires=requirements,
)