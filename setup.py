from setuptools import setup, find_packages
# 先に README.md を UTF-8(または UTF-8-SIG) で読み込んでおく
with open('README.md', 'r', encoding='utf-8-sig') as f:
    long_description = f.read()
setup(
    name='mortm',
    version='4.5b2',
    author='Nagoshi Takaaki',
    author_email='nagoshi@kthrlab.jp',
    description='旋律生成、コード推定、マルチタスクな音楽生成を行うライブラリ',
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
