# FUS network

This repository contains a simplistic network model to run simulations of FUS cortical neuromodulation, as reported in the Nature BME paper by Estrada et al. (2025). 

## Installation

- Downaload and install an [Anaconda](https://www.anaconda.com/download/) distribution
- Create a new Anaconda environment with a Python >= 3.11, and activate it:
```
conda create -n fusnetwork python=3.11
conda activate fusnetwork
```
- Clone this repository and enter its root directory:
```
git clone https://github.com/tjjlemaire/fusnetwork.git
cd fusnetwork
```
- Install the required packages:
```
pip install -r requirements.txt
```
- That's it!

## Usage

To reproduce the results of Figure 5 of the paper, execute the `hTUS_paper.ipynb` notebook.
