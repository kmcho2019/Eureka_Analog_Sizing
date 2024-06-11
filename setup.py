from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "charset-normalizer",
    "matplotlib",
    "openai",
    # 'torch<=2.0.0', # Shouldn't be necessary for this implementation as we are using jax to implement the RL training loop not issacgymenv
    'numpy==1.20.0',
    'ray>=1.1.0',
    'tensorboard>=1.14.0',
    'tensorboardX>=1.6',
    'setproctitle',
    'psutil',
    'pyyaml',
    # "gym==0.23.1", # Shouldn't be necessary for this implementation as we are using a separate environment using gymnax with its own gym>=0.26 version
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "pyvirtualdisplay",
]

# Installation operation
setup(
    name="eureka",
    author="Jason Ma",
    version="1.0",
    description="Eureka",
    keywords=["llm", "rl"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
)

