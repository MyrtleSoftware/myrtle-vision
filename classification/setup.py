import setuptools

setuptools.setup(
    name="myrtle-vision",
    version="0.0.1",
    packages=['vit_pytorch', 'utils'],
    package_dir={'': '.'},
    python_requires='>=3.7',
)
