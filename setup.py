"""
Setup script for Complexity Physics Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="complexity-physics-framework",
    version="1.0.0",
    author="Dylan Roy",
    author_email="dylan.roy@example.com",
    description="Unified physics framework from complexity minimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dylanroy/complexity-physics-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
        ],
    },
)
