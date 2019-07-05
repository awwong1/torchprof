import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchprof",
    version="0.0.1",
    author="Alexander Wong",
    author_email="alex@udia.ca",
    description="Measure neural network device specific metrics (latency, flops, etc.)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.udia.ca/alex/torchprof",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=[
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)