import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
 
# def smiles_to_morgan_fingerprint():    
calib_fraction = 0.8
    
directory = '/home/dangnd/project/cheminfo_basic/PredictionofSmallMoleculeLipophilicity/'
dfl = pd.read_csv('lipophilicity/Lipophilicity.csv')
smiles_array = dfl['smiles'].values
    
# 1d    
# generate molecues from smiles strings
list_molecules = []
for s in smiles_array:
    list_molecules.append(Chem.MolFromSmiles(s))
                
# generate fingeprints: Morgan fingerprint with radius 2
list_fingerprints = []
for molecule in list_molecules:
    list_fingerprints.append(AllChem.GetMorganFingerprintAsBitVect(molecule, 2))

# convert the RDKit explicit vectors into numpy arrays
# each is a binary vector of length 2048
x_fingerprints_1d = np.asarray(list_fingerprints)
x_fingerprints_2d = np.reshape(x_fingerprints_1d, 
                                (x_fingerprints_1d.shape[0], 32, 64))
        
# split data then save as numpy arrays
calib_num = int(calib_fraction*dfl.shape[0])
    
x_calib_1d = x_fingerprints_1d[:calib_num]
np.save(directory + 'x_calib_fingerprints_1d.npy', x_calib_1d)
    
x_calib_2d = x_fingerprints_2d[:calib_num]
np.save(directory + 'x_calib_fingerprints_2d.npy', x_calib_2d)
    
y_calib = dfl['exp'].values[:calib_num]
np.save(directory + 'y_calib.npy', y_calib)
        
x_prod_1d = x_fingerprints_1d[calib_num:]
np.save(directory + 'x_prod_fingerprints_1d.npy', x_prod_1d)
    
x_prod_2d = x_fingerprints_2d[calib_num:]
np.save(directory + 'x_prod_fingerprints_2d.npy', x_prod_2d)
    
y_prod = dfl['exp'].values[calib_num:]
np.save(directory + 'y_prod.npy', y_prod)