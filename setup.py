from setuptools import setup, find_packages

setup(
    name="base_pipeline_v2",
    version="0.0.2",
    packages=find_packages(),
    install_requires=[],  # List dependencies here
    description="Our base pipeline for ICA",
    author="ICA team",
    author_email="aantoinetteadrien@mgh.harvard.edu",
    url="https://github.com/AdrienAntoinette/base_pipeline_v2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
