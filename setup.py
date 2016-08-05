"""
Setup module
"""


from setuptools import setup


setup(
    name='basicnnets',
    version='0.1.dev1',

    description=("Different Neural Network designs implemented on top of "
                 "lasagne and Theano. Refer to README for installation details"),

    url="https://github.com/grayfall/basicnnets.git",

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
    keywords=("basic artificial neural network classes, dense, convolutional "
              "and recurrent networks"),

    packages=["nnet"],

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    install_requires=["numpy>=1.11.0",
                      "scipy>=0.17.0",
                      "multiprocess>=0.70.4"],

)
