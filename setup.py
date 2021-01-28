import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepsynth-gitract", # Replace with your own username
    version="0.0.1",
    author="Vajira Thambawita",
    author_email="vlbthambawita@gmail.com",
    description="Deepsynth gastrointestinal tract image generator.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlbthambawita/deepsynth-gitract",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'tqdm',
        'torch',
        'pandas',
        'pathlib',
        'stylegan2_pytorch',

  ],
)