# causalign/setup.py
from setuptools import find_packages, setup

setup(
    name="causalign",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core ML/Data Science
        "torch>=2.0.0",
        "numpy>=1.21.3",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.3",
        "statsmodels>=0.13.0",
        # Plotting
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "tueplots>=0.0.4",
        # LLM APIs
        "openai>=1.0.0",
        "google-generativeai>=0.3.0",
        "anthropic>=0.7.0",
        "python-dotenv>=0.19.0",
        # Development/Testing
        "pytest>=6.2.5",
        "jupyter>=1.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.10.0",
            "pylint>=2.15.0",
            "pytest-cov>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    python_requires=">=3.8,<3.14",
    author="HMD",
    author_email="hmd8142@nyu.edu",
    description="Comparing Causal Reasoning in LLMs and Humans",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hmd101/causalign",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
