from setuptools import setup, find_packages

setup(
    name="seismotrack",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "h5py>=3.8.0",
    ],
)
