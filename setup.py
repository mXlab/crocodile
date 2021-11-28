import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygan",
    version="0.1",
    author="Hugo Berard",
    author_email="berard.hugo@gmail.com",
    description="A pytorch GAN library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/a3lab/crocodile",
    project_urls={
        "Bug Tracker": "https://github.com/a3lab/crocodile/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['pygan', 'pygan.dataset', 'pygan.utils'],
    python_requires=">=3.6",
    install_requires=['simple_parsing', 'click', 'requests', 'tqdm', 'pyspng', 'ninja', 'imageio-ffmpeg==0.4.3',
                      'google-api-python-client', 'google-auth-httplib2', 'google-auth-oauthlib', 'omegaconf', 'psutil', 'scipy']
)
