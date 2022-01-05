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
    #"tensorflow_gpu==2.3.0",
    #"pypaz",
    "click",
    "pandas",
    "pycocotools",
    "xlsxwriter",
    "xlrd==1.2.0",
    "matplotlib",
    "lime",
    "shap",
    "gin-config==0.3.0",
    "jupyter",
    "psutil",
    "memory_profiler",
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
            "dext_explainer=dext.explainer.explainer:explainer",
            "dext_evaluator=dext.evaluator.evaluator:evaluator",
            "dext_error_analyzer="
            "dext.error_analyzer.error_analyzer:error_analyzer",
            "dext_visualizer=dext.visualizer.visualizer:visualizer",
            "dext_trust_analyzer="
            "dext.trust_analyzer.trust_analyzer:trust_analyzer",
        ]
    ),
    data_files=[
        (
            "dext_config",
            [
                "config/explainer.gin",
                "config/evaluator.gin",
                "config/error_analyzer.gin",
                "config/trust_analyzer.gin",
            ],
        )
    ],
    python_requires=">=3.6,<=3.9",
)