import numpy as np 
import pandas as pd 
import os 
submission_name = 'final_submission'
# create submission path 
if not os.path.exists('./submission'):
    os.mkdir('./submission')
# read results 
tar1 = pd.read_csv('./submission/submission_MMoE_sharechat_x1.csv',sep='\t').sort_values('RowId').reset_index(drop=True) 
tar2 = pd.read_csv('./submission/submission_MMoE_sharechat_x2.csv',sep='\t').sort_values('RowId').reset_index(drop=True)  
tar3 = pd.read_csv('./submission/submission_MMoE_sharechat_x3v1.csv',sep='\t').sort_values('RowId').reset_index(drop=True)  
tar4 = pd.read_csv('./submission/submission_MMoE_sharechat_x3v2.csv',sep='\t').sort_values('RowId').reset_index(drop=True) 
tar5 = pd.read_csv('./submission/submission_MMoE_sharechat_x3v3.csv',sep='\t').sort_values('RowId').reset_index(drop=True)  
tar6 = pd.read_csv('./submission/submission_MMoE_sharechat_x3v4.csv',sep='\t').sort_values('RowId').reset_index(drop=True)  
final = (tar1+tar2+tar3+tar4+tar5+tar6)/6
final['RowId'] = final['RowId'].astype(int)
# save the merged final result.
final.to_csv(f'./submission/{submission_name}.csv',sep='\t',index=None)