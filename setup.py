from setuptools import find_packages, setup

setup(
    name="xanesnet",
    version="0.1.0",
    description="Theoretical simulation of X-ray spectroscopy (XS)",
    url="https://github.com/NewcastleRSE/xray-spectroscopy-ml",
    author="Professor Thomas Penfold",
    author_email="tom.penfold@ncl.ac.uk",
    license="This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details.",
    packages=find_packages(),
    package_dir={"xanesnet": "./xanesnet"},
    install_requires=[
        # TODO
    ],
)
