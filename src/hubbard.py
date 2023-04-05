import numpy as np
import itertools
import math
import random as rand
import copy
from scipy import linalg     
import os     
import sys

### generate hatom

def make_hubbard(norbs,U,J):
    kesi = -1.0
    dim = 2**(norbs)

    U_2 = U - 2*J

    H = np.zeros((dim,dim))
    V = np.zeros((dim,dim))
    V1 = np.zeros((dim,dim))
    V2 = np.zeros((dim,dim))
    V3 = np.zeros((dim,dim))
    index0 = 0
    for num in range(0,norbs + 1):
        
    # subspace basis, first get all basis whose particle number are num
        basis = []
        dim = 0
        for cn in range(0,2**(norbs)):
            if bin(cn)[2:].count('1') == num:
                dim = dim + 1
                basis.append(cn)
        # print('dimension of Hamultionian: '+str(dim))

    # construct the  Hamiltonian
        for cn in basis:
            length = len(bin(cn))-2
            str_cn = bin(cn)[2:]

            for k in range(0,norbs-length):
                str_cn = '0' + str_cn 

            str_cn = str_cn[::-1]
            # print("cn = " + str(cn) + " , str = " + str_cn)

            for j in range(0,norbs,2):
            
            # four terms fermi operators, on-site term 
                
                str_cntmp1 = list(str_cn)
                if str_cntmp1[j] == '1' and str_cntmp1[j + 1] == '1':
                    V[cn,cn] = 1.0 + V[cn,cn]

                for k in range(0,j,2):
                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '1' and str_cntmp1[k] == '1':
                        V1[cn,cn] = 1.0 + V1[cn,cn]
                        

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '1' and str_cntmp1[k + 1] == '1':
                        V1[cn,cn] = 1.0 + V1[cn,cn]
                    

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j + 1] == '1' and str_cntmp1[k] == '1':
                        V1[cn,cn] = 1.0 + V1[cn,cn]
                        
                    
                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j + 1] == '1' and str_cntmp1[k + 1] == '1':
                        V1[cn,cn] = 1.0 + V1[cn,cn]
                        

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '1' and str_cntmp1[k] == '1':
                        V2[cn,cn] = 1.0 + V2[cn,cn]
                        

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j + 1] == '1' and str_cntmp1[k + 1] == '1':
                        V2[cn,cn] = 1.0 + V2[cn,cn]
                        

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[k] == '0' and str_cntmp1[k + 1] == '1' and str_cntmp1[j + 1] == '0' and str_cntmp1[j] == '1' :
                        
                        str_cntmp1[j] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign4 = (kesi)**inv_power
                        
                        str_cntmp1[j + 1] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j + 1].count('1')
                        fsign3 = (kesi)**inv_power
                        
                        str_cntmp1[k + 1] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:k + 1].count('1')
                        fsign2 = (kesi)**inv_power
                        
                        str_cntmp1[k] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:k].count('1')
                        fsign1 = (kesi)**inv_power
                        
                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        V3[newcn,cn] =  1.0 * fsign4*fsign3*fsign2*fsign1 + V3[newcn,cn]

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[k] == '0' and str_cntmp1[k + 1] == '0' and str_cntmp1[j] == '1' and str_cntmp1[j + 1] == '1' :
                        
                        str_cntmp1[j + 1] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j + 1].count('1')
                        fsign4 = (kesi)**inv_power
                        
                        str_cntmp1[j] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign3 = (kesi)**inv_power
                        
                        str_cntmp1[k + 1] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:k + 1].count('1')
                        fsign2 = (kesi)**inv_power
                        
                        str_cntmp1[k] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:k].count('1')
                        fsign1 = (kesi)**inv_power
                        
                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        V3[newcn,cn] =  1.0 * fsign4*fsign3*fsign2*fsign1 + V3[newcn,cn]
                
    H = U*V + U_2*V1 - J*V2 - J*V3 - J*np.conjugate(np.transpose(V3))
    row = 0
    dim = 2**norbs
    U_tran = np.zeros((dim,dim))
    # trans_basis = []
    for num in range(0,norbs + 1):
        for j in range(0,dim):
            if (bin(j).count('1') == num):
                U_tran[int(row),j] = 1
                row = row + 1
                # trans_basis.append(j)
    H = U_tran@H@U_tran.conj().T
    return H

def make_kinetic(grid,norbs,t):
    kesi = -1.0
    dim = 2**(grid*norbs)


    H = np.zeros((dim,dim))
    U_tran = np.zeros((dim,dim))
    row = 0
    basis = []
    for num in range(0,grid*norbs + 1):  
    # subspace basis, first get all basis whose particle number are num
        basis.append([])
        for cn in range(0,2**(grid*norbs)):
            if bin(cn)[2:].count('1') == num:
                basis[num].append(cn)
                U_tran[int(row),cn] = 1
                row = row + 1

    for list in basis:
    # construct the  Hamiltonian
        for cn in list:
            length = len(bin(cn))-2
            str_cn = bin(cn)[2:]

            for k in range(0,norbs-length):
                str_cn = '0' + str_cn 

            str_cn = str_cn[::-1]

        ### grid
        for i in range(grid):
            for j in range(norbs):
                if i == 0:
                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '0' and str_cntmp1[norbs + j] == '1':
                        str_cntmp1[norbs + j] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:norbs + j].count('1')
                        fsign1 = (kesi)**inv_power

                        str_cntmp1[j] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign2 = (kesi)**inv_power

                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '0' and str_cntmp1[(grid - 1)*norbs + j] == '1':
                        str_cntmp1[(grid - 1)*norbs + j] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:(grid - 1)*norbs + j].count('1')
                        fsign1 = (kesi)**inv_power

                        str_cntmp1[j] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign2 = (kesi)**inv_power

                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]
                    continue

                if i == grid - 1:
                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[j] == '0' and str_cntmp1[j - norbs] == '1':
                        str_cntmp1[j - norbs] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j - norbs].count('1')
                        fsign1 = (kesi)**inv_power

                        str_cntmp1[j] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign2 = (kesi)**inv_power

                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]

                    str_cntmp1 = list(str_cn)
                    if str_cntmp1[(grid - 1)*norbs + j] == '0' and str_cntmp1[j] == '1':
                        str_cntmp1[j] = '0'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:j].count('1')
                        fsign1 = (kesi)**inv_power

                        str_cntmp1[(grid - 1)*norbs + j] = '1'
                        str_cntmp2 = ''.join(str_cntmp1)
                        inv_power = str_cntmp2[:(grid - 1)*norbs + j].count('1')
                        fsign2 = (kesi)**inv_power

                        str_cntmp1.reverse()
                        str_newcn1 = ''.join(str_cntmp1)
                        str_newcn1 = '0b' + str_newcn1
                        newcn = int(str_newcn1,2)
                        H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]
                    continue

                str_cntmp1 = list(str_cn)
                if str_cntmp1[j] == '0' and str_cntmp1[norbs + j] == '1':
                    str_cntmp1[norbs + j] = '0'
                    str_cntmp2 = ''.join(str_cntmp1)
                    inv_power = str_cntmp2[:norbs + j].count('1')
                    fsign1 = (kesi)**inv_power

                    str_cntmp1[j] = '1'
                    str_cntmp2 = ''.join(str_cntmp1)
                    inv_power = str_cntmp2[:j].count('1')
                    fsign2 = (kesi)**inv_power

                    str_cntmp1.reverse()
                    str_newcn1 = ''.join(str_cntmp1)
                    str_newcn1 = '0b' + str_newcn1
                    newcn = int(str_newcn1,2)
                    H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]

                str_cntmp1 = list(str_cn)
                if str_cntmp1[j] == '0' and str_cntmp1[j - norbs] == '1':
                    str_cntmp1[j - norbs] = '0'
                    str_cntmp2 = ''.join(str_cntmp1)
                    inv_power = str_cntmp2[:j - norbs].count('1')
                    fsign1 = (kesi)**inv_power

                    str_cntmp1[j] = '1'
                    str_cntmp2 = ''.join(str_cntmp1)
                    inv_power = str_cntmp2[:j].count('1')
                    fsign2 = (kesi)**inv_power

                    str_cntmp1.reverse()
                    str_newcn1 = ''.join(str_cntmp1)
                    str_newcn1 = '0b' + str_newcn1
                    newcn = int(str_newcn1,2)
                    H[newcn,cn] =  1.0 * fsign2*fsign1 + H[newcn,cn]

    H = t*U_tran@(H + H.conj().T)@U_tran.conj().T
    return H

### input
norbs = 2
grid = 2
U = 2.0 ; J = U/4.0
V = make_hubbard(norbs,U,J)
T = make_kinetic(grid,norbs,t=1.0)
H = V + T





