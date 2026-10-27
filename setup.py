from setuptools import setup, find_packages
# 先に README.md を UTF-8(または UTF-8-SIG) で読み込んでおく
with open('README.md', 'r', encoding='utf-8-sig') as f:
    long_description = f.read()
setup(
    name='mortm',
    version='5.0.0b3',
    author='Nagoshi Takaaki',
    author_email='nagoshi@kthrlab.jp',
    description='旋律生成、コード推定、マルチタスクな音楽生成を行うライブラリ (MORTM5 Beta)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ayato964',
    packages=find_packages(include=['mortm', 'mortm.*']),
    include_package_data=True,
    package_data={
        'mortm.video': ['native/*.m', 'README.md'],
    },
    install_requires=[
        'torch',
        'torchaudio',
        'numpy',
        'einops',
        'loralib',
        'vector-quantize-pytorch',
        'pretty_midi',
        'mido',
        'music21',
        'midi2audio',
        'soundfile',
        'fonttools',
        'matplotlib',
        'Pillow',
        'tensorboard',
        'wandb',
        'tqdm',
        'google-api-python-client',
        'google-auth-oauthlib',
        'httplib2',
    ],
    extras_require={
        'flash': ['flash_attn'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)

