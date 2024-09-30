from setuptools import setup, find_packages

setup(
    name='rlalgs',
    version=0.1,
    url='https://github.com/guialba/rlalgs',
    author='Guiulherme Albarrans Leite',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'gymnasium',
        'pygame',
    ],
    packages=find_packages()
)