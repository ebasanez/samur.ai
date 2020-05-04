import pandas as pd
import DatasetPaths


def merge_districts(df_samur, df_districts):

	df_districts = df_districts[['DATASET_SAMUR','codigo']]
	df = pd.merge(df_samur, df_districts, left_on='Distrito', right_on='DATASET_SAMUR')
	df.drop(columns=['DATASET_SAMUR'],inplace = True)
	df.rename(columns={'codigo':'Distrito_codigo'},inplace = True)
	df.sort_values(by = 'Solicitud',inplace = True);	
	return df

# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	df_samur = pd.read_csv(DatasetPaths.SAMUR)
	df_districts = pd.read_csv(DatasetPaths.DISTRICTS)
	df = merge_districts(df_samur, df_districts)
	print(df.head())
	df.to_csv(DatasetPaths.SAMUR_MERGED.format('districts'),index = False);
	