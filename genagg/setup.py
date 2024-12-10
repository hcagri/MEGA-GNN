from setuptools import setup
from setuptools import find_packages

with open("README.md") as file:
    long_desc = file.read()

long_desc = long_desc.replace('<img src="https://github.com/Acciorocketships/generalised-aggregation/blob/main/imgs/special_cases.png" alt="Special Cases" width="500"/>', '')
long_desc = long_desc.replace('<img src="https://github.com/Acciorocketships/generalised-aggregation/blob/main/imgs/plot_inverse.png" alt="Special Cases" width="500"/>','')
long_desc = long_desc.replace('<img src="https://github.com/Acciorocketships/generalised-aggregation/blob/main/imgs/dist_ops.png" alt="Distributive Operations" width="500"/>','')

setup(
    name="genagg",
    version="2.1.1",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "wandb", "scipy"],
    author="Ryan Kortvelesy",
    url="https://github.com/Acciorocketships/generalised-aggregation",
    description="A learnable, generalised aggregation module for pytorch that parametrises the space of aggregation functions.",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    readme = "README.md",
)