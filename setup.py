"""
Setup module
"""


from setuptools import setup, find_packages


setup(
    name='genetic',
    version='0.2.dev1',

    description=('A versatile distributable genetic algorithm build with '
                 'flexibility and ease of use in mind'),

    url='https://github.com/grayfall/genetic.git',

    # Author details
    author='Ilia Korvigo',
    author_email='ilia.korvigo@gmail.com',

    license="MIT",

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering ',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords=('genetic algorithm, multiprocessing, numerical optimisation,'
              'stochastic optimisation'),

    packages=find_packages('./'),  # TODO: this is not Windows-friendly

    install_requires=[
        'numpy>=1.15.4',
        'joblib>=0.13.2',
        'tqdm>=4.31.1'
    ],
)
