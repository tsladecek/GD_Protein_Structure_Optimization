# Protein Structure Optimization based on optimizing inter-residue distance constraints

A Gradient descent based approach for creating a 3D model of protein geometry with
inter-residue distances close to the provided inter-residue distance
distributions.

### Quick Tutorial

```
python3 optimize.py -d 'domain name' -dg 'distogram path' -sp 'sequence path'
```

run 

```
python3 optimize.py -h
``` 

for more options

---
**Try it out!**
```
python optimize.py -d 1lfwA03 -dp example/1lfwA03_simulated.pt -sp example/1lfwA03.fasta -i 200 -r 5 -lr 0.02 -m 0.5 -v 5 -o example/1lfwA03_result_steric.pdb -ld 0.5
```

or with steric_clashes term

```
python optimize.py -d 1lfwA03 -dp example/1lfwA03_simulated.pt -sp example/1lfwA03.fasta -i 200 -r 5 -lr 0.02 -m 0.5 -v 5 -o example/1lfwA03_result_steric.pdb -ld 0.5 -sc True
```

## Short Introduction

Prediction of protein structure is a difficult computational problem with many
approaches developed over the last couple decades. With the advent of deep
learning many researchers decided to use neural networks for predicting
inter-residue contacts, inter-residue distances or torsional angles directly.
Thanks to the great approximative power of deep neural networks a radical
improvement in these tasks was achieved.

In 2018 Deepmind developed an algorithm (AlphaFold) for prediction of protein structure
based on extracting evolutionary relationships between homologous protein
sequences. This data was used for predicting distributions of inter-residue
distances (distogram = distance histogram) together with distributions of
torsion angles and secondary structure.
These intermediate results were used for the final step: Structure realization.
For this a differentiable script of protein geometry is required which
calculates the atomic coordinates (backbone; other atoms can be inferred from 
positions of backbone atoms) from vectors of two torsion angles &theta; and
&psi; (the third torsional angle &omega; is almost always 180 \degree and so
can be treated as a constant)

### This repository

This repository contains scripts written in `pytorch` neccessary for structure optimization:

- `structure.py`
a class of protein structure built around script of differentiable protein geometry called `G`

- `geometry_tools.py`
basic linear algebra functions used in the `structure.py` script

- `optimize.py`
basic gradient descent approach to structure optimization

- `distributions.py`
functions for approximating observed distance histograms with smooth curve

### Further reading

- **Our Thesis** (done in collaboration with Andrej Baláž) contains a lot of info 
regarding input preparation, neural network training, structure realization and
a lot more.

https://drive.google.com/file/d/17fOFwYMzYEJJEuRwDFjIdjb2OPMvrph3/view?usp=sharing

- **AlphaFold paper**
https://doi.org/10.1038/s41586-019-1923-7

## Contact
Feel free to reach me through my LinkedIn account:

https://www.linkedin.com/in/tom%C3%A1%C5%A1-sl%C3%A1de%C4%8Dek-5ab35891/