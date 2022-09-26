from setuptools import setup, find_packages
setup(
    name='find_words_from_file',
    version='0.0.0.14',
    description='input words from file and save output file',
    author='HanByulLee',
    license='MIT',
    python_requires='>=3',
    packages=find_packages(),
    install_requires = ["pandas"],
)