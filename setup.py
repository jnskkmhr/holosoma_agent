from setuptools import find_packages, setup

setup(
    name="holosoma_agent",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "tensordict",
        "loguru",
        "rich",
        "wandb",
    ],
)
