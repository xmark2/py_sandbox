# Key Purposes of setup.py

- Defining Package Metadata:
- Specifies essential details about the project, such as its name, version, author, and description.
- Example:
```commandline
from setuptools import setup

setup(
    name="my_project",
    version="1.0.0",
    author="Your Name",
    description="A sample Python project",
    packages=["my_package"],
)
```
### Managing Dependencies:
- Allows specifying required dependencies that must be installed for the project to function properly.
- Example:
```commandline
setup(
    install_requires=[
        "numpy>=1.21",
        "requests",
    ]
)
```

### Facilitating Installation:
- Running installs the package and its dependencies.

```python setup.py install```

- Running allows development mode installation.

```python setup.py develop```



# Package

```commandline
python setup.py sdist bdist_wheel
```

### Creates a source distribution
```commandline
python setup.py sdist
```

### Creates a wheel distribution
```commandline
python setup.py bdist_wheel  
```