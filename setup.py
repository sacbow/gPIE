from setuptools import setup, find_packages

setup(
    name='gpie',
    version='0.1.0',
    description='Graph-based Probabilistic Inference Engine for Computational Imaging',
    author='Hajime Ueda',
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'progress': ['tqdm'],
        'visualization': ['matplotlib', 'networkx'],
        'dev': ['tqdm', 'matplotlib', 'networkx'],
    },
)
