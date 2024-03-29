from setuptools import find_packages, setup

setup(
    name="due-cate",
    version="0.0.0",
    description="Determninistic Uncertainty Estimation for Conditional Average Treatment Effects",
    long_description_content_type="text/markdown",
    url="https://github.com/anndvision/due-cate",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.1",
        "torch>=1.8.1",
        "numpy>=1.20.2",
        "scipy>=1.6.2",
        "pandas>=1.2.4",
        "pyreadr>=0.4.1",
        "gpytorch>=1.4.2",
        "seaborn>=0.11.1",
        "hyperopt>=0.2.5",
        "ray[tune]>=1.3.0",
        "matplotlib>=3.4.2",
        "tensorboard>=2.5.0",
        "torchvision>=0.9.1",
        "scikit-learn>=0.24.2",
        "pytorch-ignite>=0.4.4",
    ],
    entry_points={
        "console_scripts": ["due-cate=due_cate.application.main:cli"],
    },
)
