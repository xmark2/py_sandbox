from setuptools import setup, find_packages
import subprocess

content = subprocess.run(['pip', 'freeze'],
                         capture_output=True,
                         text=True,
                         encoding='utf-8').stdout.splitlines()

setup(
    name="my_project",
    version="0.1.0",
    description="Your project description here",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "output*"]),
    install_requires=content,
    # classifiers=[ # Metadata for PyPI (optional)
    #     "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent"
    # ]
)
