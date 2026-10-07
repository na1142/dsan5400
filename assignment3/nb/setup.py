import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()
setuptools.setup(
    name='naive_bayes',
    version='0.0.1',
    author='Naufal Alavi',
    author_email='na1142@georgetown.edu',
    description='Naive Bayes package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_require={"dev": ["pytest", "flake8", "autopep8"]},
)