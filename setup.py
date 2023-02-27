from setuptools import find_packages, setup

setup(
    name="myrtle-vision",
    version="1.0.0",
    install_requires=[
        "torch==1.11.0",
        "torchvision==0.12.0",
        "timm==0.5.4",
        "qtorch==0.3.0",
        "psutil",
        "numpy",
        "scikit-learn",
        "tqdm",
        "tensorboard",
        "scipy",
        "pycocotools",
    ],
    packages=find_packages(
        where="src",
    ),
    package_dir={"": "src"},
)
