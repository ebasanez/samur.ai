import pandas as pd
import DatasetPaths

def merge_demographics(df_samur, df_demographics):
	df = df_samur.merge(df_demographics, left_on='Distrito', right_on = 'District')
	df.drop(columns=['District'],inplace = True)
	return df

if __name__ == '__main__':	
	df_samur = pd.read_csv(DatasetPaths.SAMUR)
	df_demographics = pd.read_csv(DatasetPaths.DEMOGRAPHICS)
	df = merge_demographics(df_samur, df_demographics)	
	print(df.head())
	df.to_csv(DatasetPaths.SAMUR_MERGED.format('demographics'),index = False)