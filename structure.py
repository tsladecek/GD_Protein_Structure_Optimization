#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protein Structure Class
"""

# %%
from geometry_tools import Geometry_tools
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from Bio.PDB.Polypeptide import one_to_three
from datetime import date
from mpl_toolkits.mplot3d import Axes3D
from distributions import *

# # Angles
CNCA = torch.tensor(np.radians(122))
NCAC = torch.tensor(np.radians(111))
CACN = torch.tensor(np.radians(116))
CCACB = torch.tensor(np.radians(150)) # this is the angle between the plane
#     # where backbone atoms are and vector CACB (counter-closkwise)

# # distances
CAC = 1.52
CN = 1.33
NCA = 1.45
CACB = 1.52

# %%
class Structure(Geometry_tools):
    def __init__(self,
                 domain,
                 distogram_path,
                 seq_path,
                 random_state=1618, 
                 normal=True,
                 distance_threshold=18
                 ):
        """
        Structure Class
        
        Input:
            domain             : str, name of the domain\n
            distogram_path     : str, path to distogram (torch pt file)\n
            seq_path           : str, path to fasta sequence\n
            random_state       : int\n
            normal             : bool, whether (scaled) normal distribution should be fitted to the distograms\n
            distance_threshold : only distances (distogram) less than "distance_threshold" are used for optimization\n
            
        Methods:
            G                   : returns distance map induced from the given set of torsion angles
            G_full              : similar to G, but returns full 3D coordinates
            copy                : returns copy of Structure object
            visualize_structure : visualizes atoms with bonds in 3D space
            pdb_atom            : returns string of atom annotation following the pdb format
            pdb_coords          : returns the structure coordinates in pdb format
        """
        
        self.distogram = torch.load(distogram_path)
        
        self.domain = domain
        self.random_state = random_state
        
        self.distance_threshold = distance_threshold
        
        L = self.distogram.shape[1]
        
        torch.manual_seed(random_state)
        self.torsion = (np.pi * (torch.rand(2* (L - 1)) - 0.5)).requires_grad_() 
        
        self.normal = normal
        
        if normal:
            self.normal_params = fit_normal(self.distogram)
            # Calculate min theoretical loss
            min_th_loss = 0
            for i in range(L - 1):
                for j in range(i + 1, L):
                    mu, sigma, s = self.normal_params[0, i, j], self.normal_params[1, i, j], self.normal_params[2, i, j]
                    
                    if mu <= distance_threshold:
                        min_th_loss -= torch.log(normal_distr(mu, mu, sigma, s))
        else:
            # Calculate min theoretical loss
            min_th_loss = 0
            for i in range(L - 1):
                for j in range(i + 1, L):
                    if torch.max(self.distogram[:, i, j]) <= distance_threshold:
                        min_th_loss -= torch.log(torch.max(self.distogram[:, i, j]))

        self.min_theoretical_loss = min_th_loss.item()
            
        with open(seq_path) as f:
            f.readline()  # fasta header
            self.seq = f.readline().strip()
        
    def G(self, coords=False):
        """
        Create differentiable protein geometry
        
        Input:
            phi     : 1D torch tensor
            psi     : 1D torch tensor
            sequence: string
            
        Output:
            2D tensor of coordinates
        """
        
        phi, psi = self.torsion[:len(self.torsion) // 2], self.torsion[len(self.torsion) // 2:]
        
        dist_mat_atoms = torch.empty((len(self.seq), 3))

        # Initialize coordinates <=> place first 3 atoms in the space
        backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                       [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                       [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                      ])

        # first c beta
        if self.seq[0] == 'G':
            dist_mat_atoms[0] = backbone[1]
        else:
            dist_mat_atoms[0] = self.place_cbeta(backbone)

        i = 1
        while i < len(self.seq):   
            
            # backbone atoms
            atoms = ['N', 'CA', 'C']
            
            angles = [psi[i-1], torch.tensor(np.pi), phi[i-1]]
            
            for j in range(3):
                atom = self.calc_atom_coords(backbone, atoms[j], angles[j])
                backbone = torch.cat((backbone, atom.view(1, 3)))

            if self.seq[i] == 'G':
                dist_mat_atoms[i] = backbone[3 * i + 1]
            else:
                dist_mat_atoms[i] = self.place_cbeta(backbone[(3 * i):(3 * (i + 1))])

            i += 1
        
        if coords:
            return dist_mat_atoms
        # distance_map
        dist_map = torch.zeros((len(self.seq), len(self.seq)))

        for i in range(len(self.seq) - 1):
            for j in range(i + 1, len(self.seq)):
                dist_map[i, j] = torch.sqrt(torch.sum((dist_mat_atoms[i] - dist_mat_atoms[j]) ** 2))

        return dist_map
    
    def copy(self):
        return copy(self)

    def G_full(self):
        """
        Calculate the backbone coordinates + C-beta satisfying the input torsion angles.
        The sequence has to be inputed in order to know whether a residue is glycin or not.
        
        Input:
            phi     : 1D torch tensor
            psi     : 1D torch tensor
            sequence: string
            
        Output:
            tuple
                2D tensor of Backbone coordinates
                2D tensor of C-beta coordinates
        """
        
        with torch.no_grad():
            phi, psi = self.torsion[:len(self.torsion)//2], self.torsion[len(self.torsion)//2:]
            # Initialize coordinates <=> place first 3 atoms in the space
            backbone = torch.tensor([[0, NCA * torch.sin(np.pi - NCAC), 0],  # N
                           [NCA * torch.cos(np.pi - NCAC), 0, 0],          # CA
                           [NCA * torch.cos(np.pi - NCAC) + CAC, 0, 0],    # C
                          ])

            for i in range(len(phi)):
                atoms = ['N', 'CA', 'C']
                
                angles = [psi[i], torch.tensor(np.pi), phi[i]]
                
                for j in range(3):
                    atom = self.calc_atom_coords(backbone, atoms[j], angles[j])
                    backbone = torch.cat((backbone, atom.view(1, 3)))

            # cbeta atoms
            cbeta_coords = torch.empty((sum([i != 'G' for i in self.seq]), 3))

            it = 0
            for i in range(len(self.seq)):
                if self.seq[i] != 'G':
                    cbeta = self.place_cbeta(backbone[(i*3):(i*3+3)])
                    cbeta_coords[it] = cbeta
                    it += 1

            return backbone, cbeta_coords

    def visualize_structure(self, figsize=None, img_path=None):
        """
        Visualizes the entire structure: backbone + C-beta atoms

        In the first step generates a list of residue coordinates:
            If Glycin: only N, CA, C atoms are present.
            Else: N, CA, CB, CA, C (the CA is there twice to make the plotting easier)        
        """
        with torch.no_grad():
            backbone, cbeta_coords = self.G_full()

            fig = plt.figure(figsize=figsize)

            entire_structure = torch.empty((len(backbone) + 2 * len(cbeta_coords), 3))

            it = 0
            cb_it = 0
            for i in range(len(self.seq)):
                N, CA, C = backbone[(3 * i):(3 * i + 3)]

                if self.seq[i] != 'G':
                    CB = cbeta_coords[cb_it]
                    cb_it += 1
                    entire_structure[it:(it+5)] = torch.cat([N, CA, CB, CA, C]).view((5, 3))
                    it += 5
                else:
                    entire_structure[it:(it+3)] = torch.cat([N, CA, C]).view((3, 3))
                    it += 3      

            coords = entire_structure.data.numpy()

            ax = fig.gca(projection='3d')
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

            if img_path is not None:
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close(fig)
    
    
    def pdb_atom(self, ind, a, aa, chain, pos, xyz):
        """
        PDB file ATOM template
        Input:
            ind  : int, atom index
            a    : str, atom ('N', 'CA', 'C' or 'CB')
            aa   : char, one letter aminoacid name
            chain: char, chain id character
            pos  : aminoacid position
            xyz  : list of coordinates
        
        Output:
            atom: pdb like ATOM list
        """
        atom = 'ATOM {:>6}  {:3} {:3} {:1} {:>4}   '.format(ind + 1, a, one_to_three(aa), chain, pos + 1)
        if 'C' in a:
            last_char = 'C'
        else:
            last_char = 'N'
        atom += '{:7.3f} {:7.3f} {:7.3f} {:6.3f} {:6.3f}           {}'.format(xyz[0], xyz[1], xyz[2], 1.0, 1.0, last_char)
        return atom

    def pdb_coords(self, domain_start=0, output_dir=None, filename=None):
        """
        Coordinates in PDB format
        
        Input:
            self        : structure
            domain_start: index of domain start position
            output_dir  : path to a directory where pdb file should be stored
        Output:
            list of pdb_atom lists
        """
        if filename is None:
            filename = f'{self.domain}_pred.pdb'
        
        backbone, cbeta = self.G_full()
        seq = self.seq
        
        chain = self.domain[4]
        # Round
        backbone = np.round(backbone.data.numpy(), 4)
        cbeta = np.round(cbeta.data.numpy(), 4)

        coords_full = []

        ind = 0
        bind = 0 # backbone ind
        cbind = 0
        for i in range(len(seq)):

            # Backbone atoms
            for j, a in enumerate(['N', 'CA', 'C']):
                coords_full.append(self.pdb_atom(ind + j, a, seq[i], chain, domain_start + i, backbone[bind+j]))

            ind += 3

            # C beta atom
            if seq[i] != 'G':
                coords_full.append(self.pdb_atom(ind, 'CB', seq[i], chain, domain_start + i, cbeta[cbind]))
                cbind += 1
                ind += 1

            bind += 3
        
        if output_dir is not None:
            with open(f'{output_dir}/{filename}', 'w') as f:
                f.write('HEADER ' + str(date.today()) + '\n')
                f.write(f'TITLE Prediction of {self.domain}\n')
                f.write(f'')
                f.write('AUTHOR Tomas Sladecek\n')
                for i in range(len(coords_full)):
                    f.write(coords_full[i] + '\n')
                    
                f.write('TER\n')
                f.write('END\n')
        else:
            return coords_full
        
