import setuptools

setuptools.setup(
    name="netrep",
    version="0.0.2",
    url="https://github.com/ahwillia/netrep",

    author="Alex Williams",
    author_email="alex.h.williams@nyu.edu",

    description="Simple methods for comparing network representations.",

    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16.5',
        'scipy>=1.3.1',
        'scikit-learn>=0.21.3',
        'tqdm>=4.32.2'
    ],
    extras_require={
        'dev': [
            'pytest>=3.7'
        ]
    },
)
