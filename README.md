# Dynamical-simulation-of-origami-based-elastic-structures-using-discrete-models

The code from the paper: Maksim Sviridenko and Igor E. Berinskii. "Dynamical simulation of origami-based elastic structures using discrete models." Smart Materials and Structures (2026). The code presents a package for explicit dynamic simulations for the origami structures using a bar-and-hinge approach. The code is featured with the implementation of constraint dynamics, including SHAKE and RATTLE, and provides three examples for popular origami patterns like Z-fold, Miura, and Kresling.

## Associated Publication

If you use this software, please cite:

Maksim Sviridenko and Igor E Berinskii,
"Dynamical simulation of origami-based elastic structures using discrete models,"
Smart Materials and Structures, 2026.

DOI: 10.1088/1361-665X/ae7f56

## Features

- Bar-and-hinge formulation for origami structures
- Elastic dynamics with extensible bars
- Constraint-based dynamics using modified SHAKE
- Constraint-based dynamics using modified RATTLE

## Installation

Clone the repository:
```bash
git clone https://github.com/MSMS-LAB/ORYGDYN
```
Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Miura Pattern
```bash
python examples/Miura_Pattern.py
```
### Kresling Pattern
```bash
python examples/Kresling_Pattern.py
```
### Z-Fold Pattern
```bash
python examples/Z_Fold_Pattern.py
```