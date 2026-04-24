"""Package setup for exoplanet-ml-classifier."""

from setuptools import find_packages, setup

setup(
    name="exoplanet-ml-classifier",
    version="0.1.0",
    description=(
        "Binary classification of NASA Kepler Object of Interest candidates "
        "using five ML classifier families."
    ),
    author="Andersson David Sánchez Méndez",
    python_requires=">=3.11",
    packages=find_packages(where=".", include=["src", "src.*"]),
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.3",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.20.0",
        "umap-learn>=0.5.6",
        "imbalanced-learn>=0.12.0",
        "shap>=0.44.0",
        "joblib>=1.3.0",
    ],
)
