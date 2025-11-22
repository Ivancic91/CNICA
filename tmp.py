import setuptools

# Read the content of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the minimum versions for your dependencies
# Note: __future__ and typing are built-in and don't need to be listed.
REQUIRED_PACKAGES = [
    "numpy>=1.18.0", 
    "scipy>=1.4.0", 
    "scikit-learn>=0.22.0" # Use scikit-learn for 'sklearn'
]

# Get a default version for the package (you should update this)
try:
    # Try to read version from the package itself
    from src.cnica._version import __version__ as VERSION
except ImportError:
    # Fallback version if the package isn't built yet
    VERSION = "0.0.1"


setuptools.setup(
    name="cnica",
    version=VERSION,
    author="Robert J. S. Ivancic",
    author_email="robert.ivancic@nist.gov",
    description="Coupled Nonnegative Independent Component Analysis (CNICA) " \
                "package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ivancic91/CNICA",
    
    # Indicates that the package contents are located in a subdirectory
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    
    # List of external dependencies
    install_requires=REQUIRED_PACKAGES,

    # Classifiers help users find your project on PyPI
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)