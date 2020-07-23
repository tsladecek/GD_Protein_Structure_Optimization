#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

# Angles
CNCA = torch.tensor(np.radians(122))
NCAC = torch.tensor(np.radians(111))
CACN = torch.tensor(np.radians(116))
CCACB = torch.tensor(np.radians(150)) # this is the angle between the plane
    # where backbone atoms are and vector CACB (counter-closkwise)

# distances
CAC = 1.52
CN = 1.33
NCA = 1.45
CACB = 1.52

class Geometry_tools:
    #def __init__(self):
    #    return
    
    def cross_product(self, k, v):
        # definition of cross product
        cp = torch.tensor([
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0]
        ])
        return cp
    
    def calc_v(self, coords, atom):
        """
        Calculate vector in the plane of previous 3 atoms in the direction
        of the target atom
        
        Input:
            coords: a 2D torch tensor of shape (L, 3)
            atom  : a string ('C', 'N' or 'CA')
        
        Output:
            vector "v": 1D torch tensor 
        """
        
        if atom == 'N':
            v_size = CN
            angle = CACN + NCAC - np.pi
        elif atom == 'CA':
            v_size = NCA
            angle = CACN + CNCA - np.pi
        elif atom == 'C':
            v_size = CAC
            angle = CNCA + NCAC - np.pi

        k = coords[-1] - coords[-2]
        k = k / torch.sqrt(torch.sum(k ** 2))

        v0 = coords[-3] - coords[-2]
        v0 = v0 / torch.sqrt(torch.sum(v0 ** 2))

        n = self.cross_product(v0, k)
        n = n / torch.sqrt(torch.sum(n ** 2))

        return v_size * self.rodrigues(v0, n, angle)
    
    def rodrigues(self, v, k, angle):
        """
        Rotate vector "v" by a angle around basis vector "k"
        see: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        
        My Implementation - commented is official rodrigues formula
        
        Input:
            v    : a 1D torch tensor
            k    : a 1D unit torch tensor
            angle: an angle in radians as a torch tensor
        
        Output:
            rotated vector: 1D torch tensor
        """

        # redefine axis system to 3 new axes with unit vectors n, k and m
        n = self.cross_product(k, v)
        n = n / torch.sqrt(torch.sum(n ** 2))

        m = self.cross_product(n, k)
        m = m / torch.sqrt(torch.sum(m ** 2))

        kv = torch.sum(k * v)
        mv = torch.sum(m * v)

        v_s = torch.sqrt(torch.sum(v ** 2))

        k_axis = k * kv
        n_axis = n * torch.sin(angle) * mv
        m_axis = m * torch.cos(angle) * mv

        vrot = k_axis + n_axis + m_axis
        
        return vrot

    def calc_atom_coords(self, coords, atom, angle):
        """
        Calculate coordinates of a new atom given list of coordinates of at least 
        previous three atoms
        
        Input:
            coords: a 2D torch tensor of shape (L, 3)
            atom  : a string ('C', 'N' or 'CA')
            angle: an angle in radians as a torch tensor
        
        Output:
            rotated and translated vector: 1D torch tensor
        """
        
        k = coords[-1] - coords[-2]
        k = k / torch.sqrt(torch.sum(k ** 2))  # unit vector

        v = self.calc_v(coords, atom)

        rotated = self.rodrigues(v, k, angle)
        return rotated + coords[-1]
    
    def place_cbeta(self, residue):
        """
        Calculate coordinates of C-beta atom. 
        
        Input:
            residue: a 2D torch tensor of coordinates in the order N, CA, C
        
        Output
            coordinates of the C-beta atom: 1D torch tensor
        """

        v1 = residue[0] - residue[1]  # vector CA-N
        v2 = residue[2] - residue[1]  # vector CA-C

        v1_scaled = CAC * (v1 / torch.sqrt(torch.sum(v1 ** 2)))

        n = v2 - v1_scaled
        n = n / torch.sqrt(torch.sum(n ** 2))

        k = self.cross_product(v2, v1)
        k = k / torch.sqrt(torch.sum(k ** 2))

        return self.rodrigues(k, n, CCACB) * CACB + residue[1]
    
