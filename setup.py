import os
from setuptools import find_packages, setup


__version__ = "0.1"

if "VERSION" in os.environ:
    BUILD_NUMBER = os.environ["VERSION"].rsplit(".", 1)[-1]
else:
    BUILD_NUMBER = os.environ.get("BUILD_NUMBER", "dev")

dependencies = [
    "numpy",
    "opencv-python",
    "tensorflow",
    "pypaz",
    "click"
]

setup(
    name="dext",
    version="{0}.{1}".format(__version__, BUILD_NUMBER),
    description="A package to explain object detectors",
    author="Deepan Chakravarthi Padmanabhan",
    install_requires=dependencies,
    packages=find_packages(),
    zip_safe=False,
    entry_points=dict(
        console_scripts=[
            "dext_explainer=dext.explainer:explainer",
        ]
    ),
    python_requires=">=3.6,<=3.9",
)