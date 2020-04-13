import numpy as np
import pandas as pd
    

directory = '/home/dangnd/project/cheminfo_basic/PredictionofSmallMoleculeLipophilicity/'
dfl = pd.read_csv('lipophilicity/Lipophilicity.csv')
    
data = dfl['exp'].values
percentile_start=0.1
percentile_step=0.1   
percentiles = np.arange(percentile_start, 1, percentile_step)
header = []
for p in percentiles:
    name = str(int(100.01*p))+'th Percentile'
    header.append(name)
        
header.append('Minimum')
header.append('Maximum')
header.append('Mean')
header.append('Standard Deviation')
header.append('Variance')
header.append('Standard Error')
header.append('# of Points')
    
list_results = list(np.quantile(data, percentiles))
list_results.append(np.min(data))
list_results.append(np.max(data))
list_results.append(np.mean(data))
list_results.append(np.std(data))
list_results.append(np.var(data))
list_results.append(np.std(data)/np.sqrt(len(data)))
list_results.append(len(data))
    
df_results = (pd.DataFrame(data = list_results, index = header)).T
df_results.to_csv('target_statistics.csv', index=False)