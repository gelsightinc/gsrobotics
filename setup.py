from setuptools import setup, find_packages

setup(
    name='gelsight',
    version='4.0',
    python_requires='>=3.8',
    packages=find_packages(),
    license='',
    author='gelsight',
    author_email='info@gelsight.com',
    install_requires=[
        'numpy>=1.17.4',
        'open3d>=0.12.0',
        'Pillow>=9.5.0',
        'pygrabber>=0.1',
        'pyusb>=1.2.1',
        'scikit_image>=0.18.3',
        'scipy>=1.10.1',
        'setuptools>=45.2.0',
        'torch>=1.11.0'
    ],
    description='A toolkit to interface with GelSight tactile sensors'
)
