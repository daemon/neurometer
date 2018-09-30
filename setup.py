import setuptools


setuptools.setup(
    name="neurometer",
    version="0.0.1",
    author="Ralph Tang",
    author_email="r33tang@uwaterloo.ca",
    description="Fine-grained neural network cost analysis tool",
    install_requires=["fire"],
    url="https://github.com/daemon/neurometer",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
