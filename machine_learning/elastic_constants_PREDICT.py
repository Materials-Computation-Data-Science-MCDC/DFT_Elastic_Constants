# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 10:56:33 2022

@author: Nathan Linton
"""

"""
This code reads in saved models from the Elastic Constants project. 
It will read in a POSCAR file and predict a Cohesive Energy for that 
POSCAR and then predict the elastic constants of the POSCAR structure.

As of right now, the POSCAR file needs to be fully relaxed (i.e. it needs
to be a CONTCAR file) containing only 16 atoms in the Ni-Cu-Au-Pd-Pt system.
"""


# Import all packages/libraries needed (will need to install several packeges)

import pandas as pd    # imports pandas, a data science package
import numpy as np     # imports numpy, a math and matrix package
import pickle
import subprocess as sp
import sys
import os


# OVITO packges
from ovito.io import * #import_file, export_file
from ovito.modifiers import * #SelectTypeModifier, DeleteSelectedModifier, CreateBondsModifier
# from ovito.modifiers import CommonNeighborAnalysisModifier 
from ovito.data import *


# @numba.jit
# def inv_nla_jit(A):
  # return np.linalg.inv(A)


# --------------------------------------------------------- #


# Individual values for average and rule of mixtures values
ind_E_vrh = {'Ni': 242.7705232, 'Cu': 160.6921993,'Au': 80.222717,'Pd': 123.3284, "Pt": 162.2530}
ind_G_V= {'Ni': 100.369536, 'Cu': 68.593602,'Au': 29.860970,'Pd': 50.401926, "Pt": 58.646028}
ind_G_R = {'Ni': 87.36708257, 'Cu': 54.0474353,'Au': 27.242851,'Pd': 39.306446, "Pt": 57.9130496}
ind_G = {'Ni': 93.868309, 'Cu': 61.3205187,'Au': 28.5519106,'Pd': 44.854186, "Pt": 58.2795388}
ind_B = {'Ni': 195.603690, 'Cu': 141.15449,'Au': 140.53044,'Pd': 164.135800, "Pt": 250.44615}
ind_nu = {'Ni': 0.293144, 'Cu': 0.3102644,'Au': 0.404857,'Pd': 0.37476998, "Pt": 0.392024}

ind_c11 = {'Ni': 274.61223, 'Cu': 185.65169,'Au': 166.75263,'Pd': 196.24294, "Pt": 318.17585}
ind_c12 = {'Ni': 156.09942, 'Cu': 118.90589,'Au': 127.41935,'Pd': 148.08223, "Pt": 216.58130}
ind_c44 = {'Ni': 127.77829, 'Cu': 92.07407,'Au': 36.65719,'Pd': 67.94964, "Pt": 63.87853}
ind_coh_energy = {'Ni': -4.89, "Cu": -3.49, 'Au':-3.04,"Pd":-3.75,"Pt":-5.54}


# From https://www.knowledgedoor.com/2/elements_handbook/palladium.html, the molar volumes
vol = {"Ni": 6.59, "Cu": 7.11, "Au": 10.21, "Pd": 8.56, "Pt": 9.09}

# USE TO CALCLATE WHEN WE ADD NEW VALUES
# c11 = ind_c11["Pt"]
# c12 = ind_c12["Pt"]
# c13 = ind_c12["Pt"]
# c22 = ind_c11["Pt"]
# c23 = ind_c12["Pt"]
# c33 = ind_c11["Pt"]
# c44 = ind_c44["Pt"]
# c55 = ind_c44["Pt"]
# c66 = ind_c44["Pt"]


# G_V = ((c11 + c22 + c33) - (c12 + c23 + c13) + 3 * (c44 + c55 + c66))/15
# B = ((c11 + c22 + c33) + 2 * (c12 + c23 + c13))/9

# Cij = np.zeros((6,6))
# Cij[0][0] = c11
# Cij[1][1] = c22
# Cij[2][2] = c33
# Cij[0][1] = c12
# Cij[0][2] = c13
# Cij[1][2] = c23
# Cij[1][0] = c12
# Cij[2][0] = c13
# Cij[2][1] = c23
# Cij[3][3] = c44
# Cij[4][4] = c55
# Cij[5][5] = c66



# Sij = np.linalg.inv(Cij)
            
# s11 = (Sij[0][0] + Sij[1][1] + Sij[2][2])/3
# s12 = (Sij[0][1] + Sij[0][2] + Sij[1][2])/3
# s44 = (Sij[3][3] + Sij[4][4] + Sij[5][5])/3

# G_R = 15/(4 * (Sij[0][0] + Sij[1][1] + Sij[2][2]) - 4 * (Sij[0][1] + Sij[1][2] + Sij[0][2]) \
        # + 3 * (Sij[3][3] + Sij[4][4] + Sij[5][5]))
# G = (G_R + G_V)/2
# nu = (3*B - 2*G)/(6*B + 2*G)

# E_VRH = (9*B*G)/(3*B + G)


# print("E_VRH",E_VRH,"G_V",G_V,"G_R",G_R,"G",G,"B",B,"nu",nu)



# ---------------- Load all saved ML models --------------- #
# Cohesive Energy
model_coh_e = pickle.load(open('ml_models/model_Cohesive_energy.pkl', 'rb'))

# Elastic Constants
model_c11 = pickle.load(open('ml_models/model_c11.pkl', 'rb'))
model_c12 = pickle.load(open('ml_models/model_c12.pkl', 'rb'))
model_c13 = pickle.load(open('ml_models/model_c13.pkl', 'rb'))
model_c23 = pickle.load(open('ml_models/model_c23.pkl', 'rb'))
model_c22 = pickle.load(open('ml_models/model_c22.pkl', 'rb'))
model_c33 = pickle.load(open('ml_models/model_c33.pkl', 'rb'))
model_c44 = pickle.load(open('ml_models/model_c44.pkl', 'rb'))
model_c55 = pickle.load(open('ml_models/model_c55.pkl', 'rb'))
model_c66 = pickle.load(open('ml_models/model_c66.pkl', 'rb'))

# Moduli
model_E_VRH = pickle.load(open('ml_models/model_E_VRH.pkl', 'rb'))
model_G_V = pickle.load(open('ml_models/model_G_V.pkl', 'rb'))
model_G_R = pickle.load(open('ml_models/model_G_R.pkl', 'rb'))
model_G = pickle.load(open('ml_models/model_G.pkl', 'rb'))
model_B = pickle.load(open('ml_models/model_B.pkl', 'rb'))

# Poisson ratio
model_nu = pickle.load(open('ml_models/model_nu.pkl', 'rb'))

# --------------------------------------------------------- #


try:
    # ----- Obtain bond iformation from user input POSCAR ----- #
    bond_info = []
    avg_bond = []
    user_file_input_1 = sys.argv[1]
    user_file_input = sys.argv[1].split('/')
    user_file_input = user_file_input[-1]
    # print(user_file_input)
    pipeline = import_file(user_file_input_1)



    # --------- CHECK IF ALL FCC ---------
    # Insert a CNA modifier to determine the structural type of each atom:
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())

    # Apply the SelectTypeModifier to select all atoms of FCC and HCP type:
    pipeline.modifiers.append(SelectTypeModifier(
        operate_on = "particles",
        property = "Structure Type",
        types = { CommonNeighborAnalysisModifier.Type.FCC,}
    ))

    # The SelectTypeModifier reports the number of selected elements as an attribute:
    data = pipeline.compute()
    fcc = data.attributes['SelectType.num_selected']
    if fcc == 16:
        

        # ---- Here we are converting the file to lammps.xyz structure format to simplify sorting ----
        file_name = "lammps_temp.xyz"
        export_file(pipeline,file_name,'xyz', columns = ["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])



        # ---- Use this to sort the atom positions to keep atom info ----
        # allows user to input the timestep*.dump from perfect case i.e. put "python sort_by_index ti" press tab and run
        readFilename = file_name


        # Choose your output file name, I suggest something different than the input while tesing, otherwise, just use file_name from above
        # outFilename = "test.xyz"
        outFilename = file_name



        # ---- Read the lammps format, and get the number of atoms, the second line (see .xyz output file format), and the atom positions
        # read coordinates using readlines()
        file1 = open(readFilename, 'r')
        Lines = file1.readlines()
        file1.close()
        num_atoms = int(Lines[0])   # number of atoms in structure
        info = str(Lines[1].strip())    # second line in the .xyz file
        coord = info.split('"')
        coord = coord[1].split(' ')
        coordinates = [coord[0:3], coord[3:6], coord[6:9]]
        coordinates =  [[float(j) for j in i] for i in coordinates] # list of lists of the x, y, and z box dimensions from poscar


        # This stores the lines containing the atom positions. I would print to make sure it has all atoms. If not, either increase or decrease
        # the number after tail -n +_. For example, it should be the 3rd line, but it might not be, it could be the fourth, so make the 3 a 4. 
        lines = sp.check_output("tail -n +3 "+ readFilename, shell=True).splitlines()

        # This just makes a list of lists containing all the atom information denoted below
        atoms = []
        for line in lines:
          l = line.split()
          atoms.append([
            int(l[0]),   # atom index
            str(l[1].decode(sys.stdout.encoding)),   # atom type
            float(l[2]), # x coordinate
            float(l[3]), # y coordinate
            float(l[4]), # z coordinate 
          ])


        atoms = pd.DataFrame(atoms,columns =['id','type','x','y','z'])


        # ---- Change the periodically transposed atoms back to where they should be ----

        for i,r in atoms.iterrows():
            if atoms.loc[i,'x'] > coordinates[0][0]-0.2:
                atoms.loc[i,'x'] = atoms.loc[i,'x']-coordinates[0][0]
                atoms.loc[i,'y'] = atoms.loc[i,'y']-coordinates[0][1]
                atoms.loc[i,'z'] = atoms.loc[i,'z']-coordinates[0][2]
            if atoms.loc[i,'y'] > coordinates[1][1]-0.2:
                atoms.loc[i,'x'] = atoms.loc[i,'x']-coordinates[1][0]
                atoms.loc[i,'y'] = atoms.loc[i,'y']-coordinates[1][1]
                atoms.loc[i,'z'] = atoms.loc[i,'z']-coordinates[1][2]
            if atoms.loc[i,'z'] > coordinates[2][2]-0.2:
                atoms.loc[i,'x'] = atoms.loc[i,'x']-coordinates[2][0]
                atoms.loc[i,'y'] = atoms.loc[i,'y']-coordinates[2][1]
                atoms.loc[i,'z'] = atoms.loc[i,'z']-coordinates[2][2]


        atoms3 = atoms.round(0).sort_values(by='y').sort_values(by = ['z','x'], ascending=[True,True])     # example, sorts by x and y coordinate values
        atoms = atoms.reindex(atoms3.index.tolist())
        atoms = atoms.values.tolist()

        # ---- write to new file with same .xyz file format using "atoms" info -----

        f = open(outFilename, 'w')
        f.write('%d \n' %num_atoms)
        f.write("%s \n" %info)

        i = 1
        for atom in atoms:
            f.write('%d %s %f %f %f \n' %(i, atom[1], atom[2], atom[3], atom[4])) 
            i += 1
        f.close()



        # Once this is done, then you can get the bond lengths again to see if the order has changed

        # This uses OVITO to import the structure file (e.g. CONTCAR, POSCAR, etc)
        pipeline = import_file(outFilename)


        # Create bonds with cutoff distance = X angstroms . (YOU DON'T NEED THIS NECESSARILY, BUT COULD USE TO SEE HOW BONDS ARE MADE)
        cut = 3.0
        pipeline.modifiers.append(CreateBondsModifier(cutoff = cut))



        # ---- This calculates all the bonds within the cutoff distance previously defined ----
        # For demonstration purposes, lets here define a compute modifier that calculates the length 
        # of each bond, storing the results in a new bond property named 'Length'.
        pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds', output_property='Length', expressions=['BondLength']))


        # ---- This puts the bond lengths into a pandas dataframe ----  
        # Let OVITO's data pipeline do the heavy work.
        # Obtain pipeline results.

        data = pipeline.compute()
        positions = data.particles.positions  # array with atomic positions
        bond_topology = data.particles.bonds.topology  # array with bond topology
        bond_lengths = data.particles.bonds['Length']     # array with bond lengths
        # print(bond_lengths)

        df = pd.DataFrame(bond_topology)
        df['r'] = bond_lengths

        if len(df)<96:
            while len(df) < 96:
                pipeline.modifiers.append(CreateBondsModifier(cutoff = cut))


                # For demonstration purposes, lets here define a compute modifier that calculates the length 
                # of each bond, storing the results in a new bond property named 'Length'.
                pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds', output_property='Length', expressions=['BondLength']))


                # Let OVITO's data pipeline do the heavy work.
                # Obtain pipeline results.
                data = pipeline.compute()
                positions = data.particles.positions  # array with atomic positions
                bond_topology = data.particles.bonds.topology  # array with bond topology
                bond_lengths = data.particles.bonds['Length']     # array with bond lengths
                # print(bond_lengths)

                df = pd.DataFrame(bond_topology)
                df['r'] = bond_lengths
                # df = df.round({'r': 5})
                cut += 0.01
                cut = round(cut,2)
            # print('Too few bonds!!',os.getcwd())
        elif len(df)>96:        
            while len(df) > 96:
                pipeline.modifiers.append(CreateBondsModifier(cutoff = cut))


                # For demonstration purposes, lets here define a compute modifier that calculates the length 
                # of each bond, storing the results in a new bond property named 'Length'.
                pipeline.modifiers.append(ComputePropertyModifier(operate_on='bonds', output_property='Length', expressions=['BondLength']))


                # Let OVITO's data pipeline do the heavy work.
                # Obtain pipeline results.
                data = pipeline.compute()
                positions = data.particles.positions  # array with atomic positions
                bond_topology = data.particles.bonds.topology  # array with bond topology
                bond_lengths = data.particles.bonds['Length']     # array with bond lengths
                # print(bond_lengths)

                df = pd.DataFrame(bond_topology)
                df['r'] = bond_lengths
                # df = df.round({'r': 5})
                cut += -0.01
                cut = round(cut,2)
            # print('Too many bonds!!',os.getcwd())


        df = df.sort_values(by=[0,1],ignore_index=True)


        # print('number of bonds is', len(df))
        file1 = open(outFilename, 'r')
        Lines = file1.readlines()
        file1.close()
        num_atoms = int(Lines[0])   # number of atoms in structure
        info = str(Lines[1].strip())    # second line in the .xyz file
        coord = info.split('"')
        coord = coord[1].split(' ')
        coordinates = [coord[0:3], coord[3:6], coord[6:9]]
        coordinates =  [[float(j) for j in i] for i in coordinates] # list of lists of the x, y, and z box dimensions from poscar


        # This stores the lines containing the atom positions. I would print to make sure it has all atoms. If not, either increase or decrease
        # the number after tail -n +_. For example, it should be the 3rd line, but it might not be, it could be the fourth, so make the 3 a 4. 
        lines = sp.check_output("tail -n +3 "+ outFilename, shell=True).splitlines()

        # This just makes a list of lists containing all the atom information denoted below
        atoms = []
        for line in lines:
          l = line.split()
          atoms.append([
            int(l[0]),   # atom index
            str(l[1].decode(sys.stdout.encoding)),   # atom type
            float(l[2]), # x coordinate
            float(l[3]), # y coordinate
            float(l[4]), # z coordinate 
          ])


        atoms = pd.DataFrame(atoms,columns =['id','type','x','y','z'])

        


        for i,r in df.iterrows():
            if atoms['type'][df[0][i]] == 'Cu':
                df.loc[[i],'Atom 1'] = int(0)
            elif atoms['type'][df[0][i]] == 'Ni':
                df.loc[[i],'Atom 1'] = int(1)
            elif atoms['type'][df[0][i]] == 'Au':
                df.loc[[i],'Atom 1'] = int(2)
            elif atoms['type'][df[0][i]] == 'Pd':
                df.loc[[i],'Atom 1'] = int(3)
            elif atoms['type'][df[0][i]] == 'Pt':
                df.loc[[i],'Atom 1'] = int(4)
            if atoms['type'][df[1][i]] == 'Cu':
                df.loc[[i],'Atom 2'] = int(0)
            elif atoms['type'][df[1][i]] == 'Ni':
                df.loc[[i],'Atom 2'] = int(1)
            elif atoms['type'][df[1][i]] == 'Au':
                df.loc[[i],'Atom 2'] = int(2)
            elif atoms['type'][df[1][i]] == 'Pd':
                df.loc[[i],'Atom 2'] = int(3)
            elif atoms['type'][df[1][i]] == 'Pt':
                df.loc[[i],'Atom 2'] = int(4)



        final = [None]*(3*len(df))

        bond_r = [None]*(len(df))

        for i,r in df.iterrows():
            bond_r = df['r'][i]
            final[i] = str(df['r'][i])
            final[i+len(df)] = str(df['Atom 1'][i])
            final[i+2*len(df)] = str(df['Atom 2'][i])


        avg_bond.append([np.mean(bond_r)])

        bond_info.append(final)
        
        
        
        
        # Calculate the averages as well as the rule of mixtures
        count_Ni = 0
        count_Cu = 0
        count_Au = 0
        count_Pd = 0
        count_Pt = 0

        for i,r in atoms.iterrows():
            if r['type'] == 'Cu':
                count_Cu += 1
            elif r['type'] == 'Ni':
                count_Ni += 1
            elif r['type'] == 'Au':
                count_Au += 1
            elif r['type'] == 'Pd':
                count_Pd += 1
            elif r['type'] == 'Pt':
                count_Pt += 1
        
        num_atoms_for_avg = count_Cu + count_Ni + count_Au + count_Pt + count_Pd
        perc_Ni = count_Ni/num_atoms_for_avg
        perc_Cu = count_Cu/num_atoms_for_avg
        perc_Au = count_Au/num_atoms_for_avg
        perc_Pd = count_Pd/num_atoms_for_avg
        perc_Pt = count_Pt/num_atoms_for_avg
        perc = {"Ni": perc_Ni, "Cu": perc_Cu, "Au": perc_Au, "Pd": perc_Pd, "Pt": perc_Pt}
        
        
        rom_upper_coh_energy = (perc['Ni']*vol['Ni']*ind_coh_energy["Ni"] + perc['Cu']*vol['Cu']*ind_coh_energy["Cu"] +\
                           perc['Au']*vol['Au']*ind_coh_energy["Au"] + perc['Pd']*vol['Pd']*ind_coh_energy["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_coh_energy["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_coh_energy = 1/((perc['Ni']*vol['Ni']*(1/ind_coh_energy["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_coh_energy["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_coh_energy["Au"]) + perc['Pd']*vol['Pd']*(1/ind_coh_energy["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_coh_energy["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
        
        rom_upper_E_VRH = (perc['Ni']*vol['Ni']*ind_E_vrh["Ni"] + perc['Cu']*vol['Cu']*ind_E_vrh["Cu"] +\
                           perc['Au']*vol['Au']*ind_E_vrh["Au"] + perc['Pd']*vol['Pd']*ind_E_vrh["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_E_vrh["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_E_VRH = 1/((perc['Ni']*vol['Ni']*(1/ind_E_vrh["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_E_vrh["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_E_vrh["Au"]) + perc['Pd']*vol['Pd']*(1/ind_E_vrh["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_E_vrh["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_G_V = (perc['Ni']*vol['Ni']*ind_G_V["Ni"] + perc['Cu']*vol['Cu']*ind_G_V["Cu"] +\
                           perc['Au']*vol['Au']*ind_G_V["Au"] + perc['Pd']*vol['Pd']*ind_G_V["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_G_V["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_G_V = 1/((perc['Ni']*vol['Ni']*(1/ind_G_V["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_G_V["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_G_V["Au"]) + perc['Pd']*vol['Pd']*(1/ind_G_V["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_G_V["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_G_R = (perc['Ni']*vol['Ni']*ind_G_R["Ni"] + perc['Cu']*vol['Cu']*ind_G_R["Cu"] +\
                           perc['Au']*vol['Au']*ind_G_R["Au"] + perc['Pd']*vol['Pd']*ind_G_R["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_G_R["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_G_R = 1/((perc['Ni']*vol['Ni']*(1/ind_G_R["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_G_R["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_G_R["Au"]) + perc['Pd']*vol['Pd']*(1/ind_G_R["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_G_R["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_G = (perc['Ni']*vol['Ni']*ind_G["Ni"] + perc['Cu']*vol['Cu']*ind_G["Cu"] +\
                           perc['Au']*vol['Au']*ind_G["Au"] + perc['Pd']*vol['Pd']*ind_G["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_G["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_G = 1/((perc['Ni']*vol['Ni']*(1/ind_G["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_G["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_G["Au"]) + perc['Pd']*vol['Pd']*(1/ind_G["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_G["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_B = (perc['Ni']*vol['Ni']*ind_B["Ni"] + perc['Cu']*vol['Cu']*ind_B["Cu"] +\
                           perc['Au']*vol['Au']*ind_B["Au"] + perc['Pd']*vol['Pd']*ind_B["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_B["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_B = 1/((perc['Ni']*vol['Ni']*(1/ind_B["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_B["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_B["Au"]) + perc['Pd']*vol['Pd']*(1/ind_B["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_B["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                              
        rom_upper_nu = (perc['Ni']*vol['Ni']*ind_nu["Ni"] + perc['Cu']*vol['Cu']*ind_nu["Cu"] +\
                           perc['Au']*vol['Au']*ind_nu["Au"] + perc['Pd']*vol['Pd']*ind_nu["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_nu["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_nu = 1/((perc['Ni']*vol['Ni']*(1/ind_nu["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_nu["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_nu["Au"]) + perc['Pd']*vol['Pd']*(1/ind_nu["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_nu["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
        
        rom_upper_c11 = (perc['Ni']*vol['Ni']*ind_c11["Ni"] + perc['Cu']*vol['Cu']*ind_c11["Cu"] +\
                           perc['Au']*vol['Au']*ind_c11["Au"] + perc['Pd']*vol['Pd']*ind_c11["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_c11["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_c11 = 1/((perc['Ni']*vol['Ni']*(1/ind_c11["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_c11["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_c11["Au"]) + perc['Pd']*vol['Pd']*(1/ind_c11["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_c11["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_c12 = (perc['Ni']*vol['Ni']*ind_c12["Ni"] + perc['Cu']*vol['Cu']*ind_c12["Cu"] +\
                           perc['Au']*vol['Au']*ind_c12["Au"] + perc['Pd']*vol['Pd']*ind_c12["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_c12["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_c12 = 1/((perc['Ni']*vol['Ni']*(1/ind_c12["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_c12["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_c12["Au"]) + perc['Pd']*vol['Pd']*(1/ind_c12["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_c12["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
                                
        rom_upper_c44 = (perc['Ni']*vol['Ni']*ind_c44["Ni"] + perc['Cu']*vol['Cu']*ind_c44["Cu"] +\
                           perc['Au']*vol['Au']*ind_c44["Au"] + perc['Pd']*vol['Pd']*ind_c44["Pd"] +\
                           perc['Pt']*vol['Pt']*ind_c44["Pt"])\
                           /(perc['Ni']*vol["Ni"] + perc['Cu']*vol["Cu"] + perc['Au']*vol["Au"] +\
                             perc['Pd']*vol["Pd"] + perc['Pt']*vol["Pt"])
        rom_lower_c44 = 1/((perc['Ni']*vol['Ni']*(1/ind_c44["Ni"]) + perc['Cu']*vol['Cu']*(1/ind_c44["Cu"]) +\
                              perc['Au']*vol['Au']*(1/ind_c44["Au"]) + perc['Pd']*vol['Pd']*(1/ind_c44["Pd"]) +\
                              perc['Pt']*vol['Pt']*(1/ind_c44["Pt"]))\
                              /(perc['Ni']*vol['Ni'] + perc['Cu']*vol['Cu'] + perc['Au']*vol['Au'] +\
                                perc['Pd']*vol['Pd'] + perc['Pt']*vol['Pt']))
        
        avg_coh_energy = (perc['Ni']*ind_coh_energy['Ni'] + perc['Cu']*ind_coh_energy['Cu'] + perc['Au']*ind_coh_energy['Au'] +\
                     perc['Pd']*ind_coh_energy['Pd'] + perc['Pt']*ind_coh_energy['Pt'])
                     
        avg_E_VRH = (perc['Ni']*ind_E_vrh['Ni'] + perc['Cu']*ind_E_vrh['Cu'] + perc['Au']*ind_E_vrh['Au'] +\
                     perc['Pd']*ind_E_vrh['Pd'] + perc['Pt']*ind_E_vrh['Pt'])
                     
        avg_G_V = (perc['Ni']*ind_G_V['Ni'] + perc['Cu']*ind_G_V['Cu'] + perc['Au']*ind_G_V['Au'] +\
                     perc['Pd']*ind_G_V['Pd'] + perc['Pt']*ind_G_V['Pt'])
                     
        avg_G_R = (perc['Ni']*ind_G_R['Ni'] + perc['Cu']*ind_G_R['Cu'] + perc['Au']*ind_G_R['Au'] +\
                     perc['Pd']*ind_G_R['Pd'] + perc['Pt']*ind_G_R['Pt'])
                     
        avg_G = (perc['Ni']*ind_G['Ni'] + perc['Cu']*ind_G['Cu'] + perc['Au']*ind_G['Au'] +\
                     perc['Pd']*ind_G['Pd'] + perc['Pt']*ind_G['Pt'])
                     
        avg_B = (perc['Ni']*ind_B['Ni'] + perc['Cu']*ind_B['Cu'] + perc['Au']*ind_B['Au'] +\
                     perc['Pd']*ind_B['Pd'] + perc['Pt']*ind_B['Pt'])
                    
        avg_nu = (perc['Ni']*ind_nu['Ni'] + perc['Cu']*ind_nu['Cu'] + perc['Au']*ind_nu['Au'] +\
                     perc['Pd']*ind_nu['Pd'] + perc['Pt']*ind_nu['Pt'])
                     
        avg_c11 = (perc['Ni']*ind_c11['Ni'] + perc['Cu']*ind_c11['Cu'] + perc['Au']*ind_c11['Au'] +\
                     perc['Pd']*ind_c11['Pd'] + perc['Pt']*ind_c11['Pt'])
                     
        avg_c12 = (perc['Ni']*ind_c12['Ni'] + perc['Cu']*ind_c12['Cu'] + perc['Au']*ind_c12['Au'] +\
                     perc['Pd']*ind_c12['Pd'] + perc['Pt']*ind_c12['Pt'])
                     
        avg_c44 = (perc['Ni']*ind_c44['Ni'] + perc['Cu']*ind_c44['Cu'] + perc['Au']*ind_c44['Au'] +\
                     perc['Pd']*ind_c44['Pd'] + perc['Pt']*ind_c44['Pt'])
                     
        
        
        os.remove(file_name)
        os.remove(user_file_input_1)
    else:
        sys.exit()


    # -------------------------------------------------- #



    # ----- Predict Cohesive energy and create the ML table for other predictions ---- #
    """
    Here we may need to leave an option for user input final energy of structure
    to calculate the cohesive energy rather than predict it, like the original
    models. (can do this with a simple if statement such as: if user provides 
    final energy in eV, do the calculation, if not, predict using this code). 
    I can provide the calculation code if needed.
    """
    
    header = []
    for i in range(0,96):
        header.append(str('%s r' %(i+1)))   
    for i in range(0,96):
        header.append(str('%s Type 1' %(i+1) ))
    for i in range(0,96):
        header.append(str('%s Type 2' %(i+1) ))

    bond_info = pd.DataFrame(bond_info)

    ml_table = pd.concat([bond_info],axis=1)
    ml_table.columns = header

    energy = model_coh_e.predict(ml_table)

    # Predict Cohesive Energy
    ml_table.insert(0,'Cohesive_energy', energy[0])

    # ---------------------------------------------------- #

    # -------- Predict elastic constants, moduli, nu --------------- #
    c11 = model_c11.predict(ml_table)
    c12 = model_c12.predict(ml_table)
    c13 = model_c13.predict(ml_table)
    c23 = model_c23.predict(ml_table)
    c22 = model_c22.predict(ml_table)
    c33 = model_c33.predict(ml_table)
    c44 = model_c44.predict(ml_table)
    c55 = model_c55.predict(ml_table)
    c66 = model_c66.predict(ml_table)
    G_V = model_G_V.predict(ml_table)
    G_R = model_G_R.predict(ml_table)
    G = model_G.predict(ml_table)
    B = model_B.predict(ml_table)
    E_VRH = model_E_VRH.predict(ml_table)
    nu = model_nu.predict(ml_table)




    columns_out = ["E_Coh [eV/atom]","C11 [GPa]","C12 [GPa]","C13 [GPa]","C22 [GPa]",\
    "C23 [GPa]","C33 [GPa]","C44 [GPa]","C55 [GPa]","C66 [GPa]","E_VRH [GPa]","G_V [GPa]",\
    "G_R [GPa]","G [GPa]","B [GPa]","nu []"]
    predicted_out = [-energy,c11,c12,c13,c22,c23,c33,c44,c55,c66,E_VRH,G_V,G_R,G,B,nu]
    predicted_out = [x for xs in predicted_out for x in xs]
    predicted_out = np.array(predicted_out)
    predicted_out = np.around(predicted_out,3)
    predicted_out = list(predicted_out)
    average_out = [-avg_coh_energy,avg_c11,avg_c12,avg_c12,avg_c11,avg_c12,avg_c11,avg_c44,avg_c44,\
                   avg_c44,avg_E_VRH,avg_G_V,avg_G_R,avg_G,avg_B,avg_nu]
    average_out = np.array(average_out)
    average_out = np.around(average_out,3)
    average_out = list(average_out)

    rom_lower_out = [-rom_lower_coh_energy,rom_lower_c11,rom_lower_c12,rom_lower_c12,rom_lower_c11,\
                     rom_lower_c12,rom_lower_c11,rom_lower_c44,rom_lower_c44,\
                     rom_lower_c44,rom_lower_E_VRH,rom_lower_G_V,rom_lower_G_R,rom_lower_G,rom_lower_B,rom_lower_nu]

    rom_upper_out = [-rom_upper_coh_energy,rom_upper_c11,rom_upper_c12,rom_upper_c12,rom_upper_c11,\
                     rom_upper_c12,rom_upper_c11,rom_upper_c44,rom_upper_c44,\
                     rom_upper_c44,rom_upper_E_VRH,rom_upper_G_V,rom_upper_G_R,rom_upper_G,rom_upper_B,rom_upper_nu]
    rom_out = pd.DataFrame([rom_lower_out,rom_upper_out]).round(3).T
    rom_out.columns = ['lower','upper']
    rom_out['RoM'] = rom_out['lower'].astype(str) +"-"+ rom_out['upper'].astype(str)
    rom_out = rom_out.RoM.values.tolist()

    header_out = ["Property","Predicted","Average","Rule of Mixtures (lower-upper)"]

    final_output = [columns_out,predicted_out,average_out,rom_out]

    if count_Ni != 0:
        header_out.append('Ni')
        final_output.append([round(-ind_coh_energy['Ni'],3),round(ind_c11['Ni'],3),round(ind_c12['Ni'],3),round(ind_c12['Ni'],3),\
                             round(ind_c11['Ni'],3),round(ind_c12['Ni'],3),round(ind_c11['Ni'],3),round(ind_c44['Ni'],3),round(ind_c44['Ni'],3),\
                             round(ind_c44['Ni'],3),round(ind_E_vrh['Ni'],3),round(ind_G_V['Ni'],3),round(ind_G_R['Ni'],3),round(ind_G['Ni'],3),\
                             round(ind_B['Ni'],3),round(ind_nu['Ni'],3)])
    if count_Cu != 0:
        header_out.append('Cu')
        final_output.append([round(-ind_coh_energy['Cu'],3),round(ind_c11['Cu'],3),round(ind_c12['Cu'],3),round(ind_c12['Cu'],3),\
                             round(ind_c11['Cu'],3),round(ind_c12['Cu'],3),round(ind_c11['Cu'],3),round(ind_c44['Cu'],3),round(ind_c44['Cu'],3),\
                             round(ind_c44['Cu'],3),round(ind_E_vrh['Cu'],3),round(ind_G_V['Cu'],3),round(ind_G_R['Cu'],3),round(ind_G['Cu'],3),\
                             round(ind_B['Cu'],3),round(ind_nu['Cu'],3)])
    if count_Au != 0:
        header_out.append('Au')
        final_output.append([round(-ind_coh_energy['Au'],3),round(ind_c11['Au'],3),round(ind_c12['Au'],3),round(ind_c12['Au'],3),\
                             round(ind_c11['Au'],3),round(ind_c12['Au'],3),round(ind_c11['Au'],3),round(ind_c44['Au'],3),round(ind_c44['Au'],3),\
                             round(ind_c44['Au'],3),round(ind_E_vrh['Au'],3),round(ind_G_V['Au'],3),round(ind_G_R['Au'],3),round(ind_G['Au'],3),\
                             round(ind_B['Au'],3),round(ind_nu['Au'],3)])
    if count_Pd != 0:
        header_out.append('Pd')
        final_output.append([round(-ind_coh_energy['Pd'],3),round(ind_c11['Pd'],3),round(ind_c12['Pd'],3),round(ind_c12['Pd'],3),\
                             round(ind_c11['Pd'],3),round(ind_c12['Pd'],3),round(ind_c11['Pd'],3),round(ind_c44['Pd'],3),round(ind_c44['Pd'],3),\
                             round(ind_c44['Pd'],3),round(ind_E_vrh['Pd'],3),round(ind_G_V['Pd'],3),round(ind_G_R['Pd'],3),round(ind_G['Pd'],3),\
                             round(ind_B['Pd'],3),round(ind_nu['Pd'],3)])
    if count_Pt != 0:
        header_out.append('Pt')
        final_output.append([round(-ind_coh_energy['Pt'],3),round(ind_c11['Pt'],3),round(ind_c12['Pt'],3),round(ind_c12['Pt'],3),\
                             round(ind_c11['Pt'],3),round(ind_c12['Pt'],3),round(ind_c11['Pt'],3),round(ind_c44['Pt'],3),round(ind_c44['Pt'],3),\
                             round(ind_c44['Pt'],3),round(ind_E_vrh['Pt'],3),round(ind_G_V['Pt'],3),round(ind_G_R['Pt'],3),round(ind_G['Pt'],3),\
                             round(ind_B['Pt'],3),round(ind_nu['Pt'],3)])

    data_out = pd.DataFrame(final_output)
    data_out = data_out.T
    data_out.columns = header_out

    # OUTPUT TO CSV FILE FORMAT FOR WEBSITE UI OUTPUT

    # ---------------------------------------------------- #

    data_out.to_csv("./output/%s" %user_file_input, index = False)

# IF ERRORS THROWN
except SystemExit: # This error will check for a system that does not contain 16 atoms
    text_file = open("./output/%s" %user_file_input, "w")
    n = text_file.write('ERROR! Your file does not contain 16 atoms in a 2x2x1 or is not an FCC structure. Please input a valid structure file and try again. (Note: In the future we plan to predict on different system sizes)')
    text_file.close()
    if os.path.isfile(user_file_input_1): 
        os.remove(user_file_input_1)
    sys.exit()
except: # This error will check for gibberish text files as well as files contianing elements not in the current model
    text_file = open("./output/%s" %user_file_input, "w")
    n = text_file.write('ERROR! Your file did not contain a valid structure, please input a valid structure file and try again. We currently only predict on structures containing Ni, Cu, Au, Pd, and/or Pt.')
    text_file.close()
    if os.path.isfile(user_file_input_1): 
        os.remove(user_file_input_1)
