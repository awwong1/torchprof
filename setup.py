import codecs
import os
import re
import setuptools

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")


long_description = read("README.md")
version = find_version("torchprof", "__init__.py")

setuptools.setup(
    name="torchprof",
    version=version,
    author="Alexander Wong",
    author_email="alex@udia.ca",
    description="Measure neural network device specific metrics (latency, flops, etc.)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/awwong1/torchprof",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=["torch>=1.1.0,<2"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
