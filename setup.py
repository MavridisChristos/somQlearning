# Author: Christos Mavridis <mavridis@umd.edu>

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="somQlearning", 
    version="0.0.1",
    author="Christos N. Mavridis",
    author_email="mavridis@umd.edu",
    description="Q-Learning with Online Self-Organizing Maps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MavridisChristos/somQlearning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)