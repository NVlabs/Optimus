# setup.py
#!/usr/bin/env python
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="optimus",
    packages=[package for package in find_packages() if package.startswith("optimus")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="Official code release for Optimus: Imitating Task and Motion Planning with Visuomotor Transformers",
    author="Murtaza Dalal",
    url="https://github.com/NVlabs/Optimus.git",
    author_email="mdalal@andrew.cmu.edu",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
