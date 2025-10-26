#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiGridDet Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="multigriddet",
    version="1.0.0",
    author="Solomon Negussie Tesema",
    author_email="solomon.negussie.tesema@gmail.com",
    description="MultiGridDet: A modern implementation of 'Multi-Grid Redundant Bounding Box Annotation for Accurate Object Detection'",
    keywords="object detection, multi-grid redundant bounding box annotation, traina",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Reports": "https://github.com/solomonnegussie/MultiGridDet/issues",
        "Source Code": "https://github.com/solomonnegussie/MultiGridDet",
        "Paper": "https://ieeexplore.ieee.org/document/9730183",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    include_package_data=True,
    package_data={
        "multigriddet": [
            "configs/*.txt",
            "configs/*.yaml",
        ],
    },
    entry_points={
        "console_scripts": [
            "multigriddet-inference=multigriddet.scripts.inference:main", # infer the model
            "multigriddet-train=multigriddet.scripts.train:main", # train the model
            "multigriddet-eval=multigriddet.scripts.eval:main", # evaluate the model
        ],
    },
)





