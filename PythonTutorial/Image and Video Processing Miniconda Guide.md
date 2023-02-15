# Image and Video Processing (with Python)

This is a guide to help you install the environment for image and video processing course. 

# Getting started

To be able to follow the tutorial and finish the computer assignment, you are going to need a laptop with Miniconda (a minimal version of Anaconda) and several Python packages installed.
The following instruction would work as is for Mac or Ubuntu Linux users, Windows users would need to install and work in the [Git BASH](https://gitforwindows.org/) terminal.

## Download and install Miniconda

Please go to the [Anaconda website](https://conda.io/miniconda.html).
Download and install *the latest* Miniconda version for *Python* 3.7 for your operating system.

```bash
wget <http:// link to miniconda>
sh <miniconda*.sh>
```

## Create isolated Miniconda environment

Change directory (`cd`) into the tutorial folder, then type:

```bash
# cd PythonTutorial
conda env create -f environment.yml
conda activate ImV
```

## Start Jupyter Notebook or JupyterLab

Start from terminal as usual:

```bash
jupyter lab
```

Or, for the classic interface:

```bash
jupyter notebook
```