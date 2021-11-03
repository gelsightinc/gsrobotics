from setuptools import setup

setup(
    name='gelsight',
    version='1.0',
    packages=['gelsight', 'gelsightcore'],
    package_data={'gelsightcore': ['*.so*']},
    license='',
    author='gelsight',
    author_email='info@gelsight.com',
    install_requires=[
        'numpy',
    ],
    description='A toolkit to interface with Gelsight tactile devices'
)
