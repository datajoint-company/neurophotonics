import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name="photonics", 
    version="0.0.1",
    author="Dimitri Yatsenko",
    author_email="dimitri@vathes.com",
    description="Neurophotonics probe simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dimitri-yatsenko/simulight",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",],
    python_requires='>=3.6',
)
