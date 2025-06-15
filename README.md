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


# Install jupyter scheduler

```commandline
pip install jupyter_scheduler
jupyter server extension enable jupyter_scheduler
jupyter lab

jupyter lab clean
jupyter lab build
```


If not works I tried these
```commandline
pip install jupyter_scheduler
pip install --upgrade jupyterlab
pip install --force-reinstall psutil==5.9.0 jupyterlab==3.1.4 anyio==4.7.0 jupyter_server==2.15.0 typing_extensions==4.10.0 websockets==10.4
```

# Run jupyter lab

```commandline
jupyter lab
```
