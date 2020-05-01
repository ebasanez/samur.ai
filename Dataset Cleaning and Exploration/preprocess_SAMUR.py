import numpy as np
import pandas as pd
import DatasetPaths
from datetime import *

def preprocess_samur(dfs_samur):
	
	#Merge all datasets
	df = pd.concat(dfs_samur)
	
	# remove cancelled request
	df = df[~df['Devuelto'] == True]
	
	# Add hour column
	df['Solicitud'] = df['Solicitud'].astype('datetime64[ns]') 
	df['Hour'] = df['Solicitud'].dt.hour
	
	# Remove unnamed columns (unused indexes)
	df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	
	# reset index and sort
	df.sort_values(by = 'Solicitud',inplace = True);	
	df.reset_index(drop = True, inplace = True)
		
	return df;
	
if __name__ == '__main__':	
	dfs_samur = [pd.read_csv(x) for x in [DatasetPaths.SAMUR_2017,DatasetPaths.SAMUR_2018,DatasetPaths.SAMUR_2019]]
	df = preprocess_samur(dfs_samur)
	print(df.shape)
	print(df.head())
	print(df.tail())
	df.to_csv(DatasetPaths.SAMUR,index=False)