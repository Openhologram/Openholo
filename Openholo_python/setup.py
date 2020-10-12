import setuptools
import os

mydir = os.path.dirname(__file__)
if mydir:
    os.chdir(mydir)

with open("README.md", "r") as fhandle:
    long_description = fhandle.read()

version = '0.0.3'

setuptools.setup(
    name="ophpy",
    version=version,
    author="YoungRok Kim",
    author_email="faller825@khu.ac.kr",
    description="Open source project about Computer Generated Hologram",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/robin9804/openholo_py",
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'pillow', 'plyfile', 'numba'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
