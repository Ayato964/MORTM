from setuptools import setup, find_packages

setup(
    name='MORTM',
    version='1.0b18',
    author='Nagoshi Takaaki',
    author_email='nagoshi@kthrlab.jp',
    description='音楽の旋律生成を実現したシステム',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Ayato964',
    packages=['mortm'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)
