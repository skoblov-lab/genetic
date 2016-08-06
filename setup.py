"""
Setup module
"""


from setuptools import setup


setup(
    name='genetic',
    version='0.1.dev2',

    description=("A versatile distributable genetic algorithm build with"
                 "flexibility and ease of use in mind"),

    url="https://github.com/grayfall/genetic.git",

    # Author details
    author="Ilia Korvigo",
    author_email="ilia.korvigo@gmail.com",

    license="MIT",

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering ",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],

    # What does your project relate to?
    keywords=("genetic algorithm, multiprocessing, numerical optimisation,"
              "stochastic optimisation"),

    packages=["genetic"],

    install_requires=["numpy>=1.11.0",
                      "scipy>=0.17.0",
                      "multiprocess>=0.70.4"],
)
