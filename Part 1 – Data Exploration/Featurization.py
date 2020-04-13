import pandas as pd
import numpy as np 


# def smiles_to_embedding():    
calib_fraction = 0.8
    
directory = '/home/dangnd/project/cheminfo_basic/PredictionofSmallMoleculeLipophilicity/'
dfl = pd.read_csv('lipophilicity/Lipophilicity.csv')
smiles_array = dfl['smiles'].values
        
# create sorted, unique array of smiles characters
array_smiles_unique = np.unique(list("".join(smiles_array)))
        
# create embedded (not padded)
mapping = dict(zip(array_smiles_unique, range(1, 1 + len(array_smiles_unique))))
list_embedded_smiles = []
for s in smiles_array:
    list_embedded_smiles.append([mapping[i] for i in s])
        
df_lipophilicity_embedded_smiles_shuffled = pd.DataFrame(index=dfl.index,
                                                            data=dfl.values,
                                                            columns=dfl.columns)
df_lipophilicity_embedded_smiles_shuffled['smiles embedded'] = list_embedded_smiles
    
    
df_lipophilicity_embedded_smiles_shuffled.to_pickle(directory +
                'df_lipophilicity_embedded_smiles_shuffled.pkl')
    
    
# split and save as numpy arrays
calib_num = int(calib_fraction*df_lipophilicity_embedded_smiles_shuffled.shape[0])
x_calib = df_lipophilicity_embedded_smiles_shuffled['smiles embedded'].values[:calib_num]
np.save(directory + 'x_calib_smiles_embedded.npy', x_calib)
x_prod = df_lipophilicity_embedded_smiles_shuffled['smiles embedded'].values[calib_num:]
np.save(directory + 'x_prod_smiles_embedded.npy', x_prod)