from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'imsolve'
LONG_DESCRIPTION = 'Low memory, matrix free slovers useful for large scale image processing'

setup(
        name="imsolve", 
        version=VERSION,
        author="Jens Renders",
        author_email="<jens.renders@uantwerpen.be>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "scipy",
            "pylops",
        ],
        
        keywords=['optimization', 'image processing', 'linear algebra'],
        classifiers= [
            "Development Status :: 2 - Pre-Alpha",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ]
)