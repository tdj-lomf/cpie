import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cpie",
    version="1.0.0",
    author="Jun Toda",
    author_email="lomf@hotmail.co.jp",
    description="An evolutionary computation algorithm named CPIE in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tdj-lomf/cpie",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)