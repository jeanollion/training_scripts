import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="training_scripts",
    version="0.0.1",
    author="Jean Ollion",
    author_email="jean.ollion@polytechnique.org",
    description="DL Training Scripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeanollion/training_scripts",
    download_url='https://github.com/jeanollion/training_scripts/archive/0.0.1.tar.gz',
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=['dataset_iterator>=0.3.4', 'pix_mclass>=0.0.3', 'distnet_2d>=0.1.1']
)
