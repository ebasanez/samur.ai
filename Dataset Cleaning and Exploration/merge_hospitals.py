import sys, getopt
import utils
import pandas as pd
import DatasetPaths

KEY = 'Hospital'
COLUMNS_TO_KEEP = ['Hospital','km0_x','km0_y']

def merge_hospitals(df_samur, df_hospitals):

	df_hospitals = df_hospitals[['name_orig','Hospital','hospital_x','hospital_y']]
	df_samur.rename(columns={'Hospital':'Hospital_old'}, inplace=True)
	df = pd.merge(df_samur, df_hospitals, left_on='Hospital_old', right_on='name_orig', how = 'outer')
	df.drop(columns=['Hospital_old','name_orig'],inplace=True)
	
	# Remove values for hospitals 'Alcalá de Henares (Ppe. de Asturias)' and 'Getafe' because those are outside Madrid
	df = df[~df.Hospital.isin(['Alcalá de Henares (Ppe. de Asturias)','Getafe'])]
	df.sort_values(by = 'Solicitud',inplace = True);	
	return df

# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	df_samur = pd.read_csv(DatasetPaths.SAMUR)
	df_hospitals = pd.read_csv(DatasetPaths.HOSPITALS)
	df = merge_hospitals(df_samur, df_hospitals)
	print(df.head())
	print(DatasetPaths.SAMUR_MERGED.format('hospitals'))
	df.to_csv(DatasetPaths.SAMUR_MERGED.format('hospitals'),index = False);
	