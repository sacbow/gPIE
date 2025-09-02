from setuptools import setup, find_packages

setup(
    name='gpie',
    version='0.1.0',
    description='Graph-based Probabilistic Inference Engine for Computational Imaging',
    author='Hajime Ueda',
    author_email='ueda@mns.k.u-tokyo.ac.',  # Optional but recommended
    python_requires='>=3.10',
    packages=find_packages(),
    install_requires=[
        'numpy>=2.2.6',
    ],
    extras_require={
        'visualization': ['bokeh>=3.7.3'],
        'dev': ['tqdm', 'bokeh>=3.7.3'],
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    include_package_data=True,
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)
