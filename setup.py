from setuptools import setup, find_packages

setup(
    name='driver-alertness-detection',
    version='0.1.0',
    description='Driver alertness detection using CNNs',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[l.strip() for l in open('requirements.txt') if l.strip()],
)
