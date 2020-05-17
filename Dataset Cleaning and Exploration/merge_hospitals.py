import sys, getopt
import utils
import pandas as pd
import DatasetPaths
import yaml

KEY = 'Hospital'
COLUMNS_TO_KEEP = ['Hospital','km0_x','km0_y']

def merge_hospitals(df_samur, df_hospitals):

	df_hospitals = df_hospitals[['name_orig','Hospital','hospital_x','hospital_y']]
	df_samur.rename(columns={'Hospital':'Hospital_old'}, inplace=True)
	df = pd.merge(df_samur, df_hospitals, left_on='Hospital_old', right_on='name_orig', how = 'outer')
	df.drop(columns=['Hospital_old','name_orig'],inplace=True)
	
	# Remove values for hospitals 'Alcalá de Henares (Ppe. de Asturias)', 'Getafe' because those are outside Madrid
	df = df[~df.Hospital.isin(['Alcalá de Henares (Ppe. de Asturias)','Getafe'])]
	df.sort_values(by = 'Solicitud',inplace = True);	
	return df

def assign_ambulances(df_samur, df_hospitals, total_ambulances):
	dfg = df_samur.groupby('Hospital').agg({'Hospital':'count'})
	dfg.rename(columns={'Hospital':'Total'}, inplace = True)
	df = pd.merge(dfg, df_hospitals, left_on='Hospital', right_on='Hospital')
	df = df[df['district_code'] != -1]
	df.reset_index(inplace = True, drop = True)
	total = df.Total.sum()
	df['Ambulances'] = round(df['Total'] / total * total_ambulances)
	df = df.astype({'Ambulances':'int32'})
	print(df)
	return df	
	
	
# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	df_samur = pd.read_csv(DatasetPaths.SAMUR)
	df_hospitals = pd.read_csv(DatasetPaths.HOSPITALS)
	df = merge_hospitals(df_samur, df_hospitals)
	print(df.head())
	df.to_csv(DatasetPaths.SAMUR_MERGED.format('hospitals'),index = False);
	
	df = assign_ambulances(df,df_hospitals,utils.NUMBER_OF_AMBULANCES)
	# Transform to dictionary and save to yaml
	df_dict = [{0:{'available_amb':0,'name':'NaN','loc':{'district_code':0,'x':0.0,'y':0.0}}}]
	for index,r in df.iterrows():
		df_dict.append({index+1:{'available_amb':r.Ambulances,'name':r.Hospital,'loc':{'district_code':r.district_code,'x':r.hospital_x,'y':r.hospital_y}}})
	yaml_file = open(DatasetPaths.HOSPITALS_YAML,"w+",encoding='utf8')
	yaml.dump(df_dict,yaml_file,allow_unicode = True)
	
	