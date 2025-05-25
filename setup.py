from setuptools import setup, find_packages
# 先に README.md を UTF-8(または UTF-8-SIG) で読み込んでおく
with open('README.md', 'r', encoding='utf-8-sig') as f:
    long_description = f.read()
setup(
    name='MORTM',
    version='4.0b6',
    author='Nagoshi Takaaki',
    author_email='nagoshi@kthrlab.jp',
    description='音楽の旋律生成を実現したシステム',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ayato964',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)
