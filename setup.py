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
    name="tamp_imitation",
    packages=[package for package in find_packages() if package.startswith("tamp_imitation")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="tamp_imitation",
    author="Murtaza Dalal",
    url="https://gitlab-master.nvidia.com/murtazad/tamp_imitation",
    author_email="",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
