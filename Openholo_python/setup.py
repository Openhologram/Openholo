import setuptools

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()
setuptools.setup(
    name="openholo python project", # Put your username here!
    version="0.0.1", # The version of your package!
    author="YoungRok Kim", # Your name here!
    author_email="faller825@khu.ac.kr", # Your e-mail here!
    description="Opensource project about Computer Generated Hologram",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/robin9804/openholo_py", # Link your package website here! (most commonly a GitHub repo)
    packages=setuptools.find_packages(), # A list of all packages for Python to distribute!
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-2.0 License",
        "Operating System :: OS Independent",
    ], # Enter meta data into the classifiers list!
    python_requires='>=3.7', # The version requirement for Python to run your package!
)
