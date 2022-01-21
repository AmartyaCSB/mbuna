from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='mbuna',
    version='0.0.1',
    description='experimental cichlid behavioral analysis suite',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tlancaster6/mbuna',
    author='Tucker J. Lancaster',
    author_email='tlancaster6@gatech.edu',
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='behavior, ethology, cv, yolov5, fiftyone',
    packages=find_packages(),
    python_requires='>=3.7, <3.10',
    install_requires=['yolov5', 'fiftyone'],
    entry_points={"console_scripts": ["mbuna=mbuna.core.cli:main"]},
)
