#!/bin/bash

dir_bin="bin/project"
dir_project="project"
dir_project_test=$dir_project"/test"
mkdir -p $dir_bin
mkdir -p $dir_project_test

touch setup.py
touch $dir_project"/__init__.py" $dir_project"/main.py"
touch $dir_project_test"/__init__.py" $dir_project_test"/test_main.py"
