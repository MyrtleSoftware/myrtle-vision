import setuptools

setuptools.setup(
    name="myrtle-vision",
    version="0.0.1",
    packages=["myrtle_vision"],
    package_dir={'': '.'},
    python_requires='>=3.7',
    install_requires=[
        'torch',
        'torchvision',
        'psutil',
        'timm',
        'qtorch',
        'numpy',
        'scikit-learn',
        'tqdm',
    ],
)
