from setuptools import setup
from setuptools import find_packages

setup(
    name='curve_fit',
    version='0.1',
    description='A curve fitting library',
    url='https://github.com/data2code/curve_fit',
    author='Yingyao Zhou',
    author_email='yingyao.zhou@gmail.com',
    license='Apache-2.0',
    zip_safe=False,
    packages=find_packages(),
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'seaborn',
                      'tqdm',
                      'tabulate',
                      ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
    ],
)
