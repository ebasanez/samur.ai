import pandas as pd
import DatasetPaths
from merge_medical_centers import merge_medical_centers
from merge_hospitals import merge_hospitals
from merge_demographics import merge_demographics
from merge_districts import merge_districts


def merge_all(df_emergencies, df_hospitals, df_medical_centers, df_demographics, df_districts):

	df = merge_hospitals(df_emergencies,df_hospitals)
	df = merge_demographics(df,df_demographics)
	df = merge_districts(df,df_districts)
	df = merge_medical_centers(df,df_medical_centers)
	return df;

if __name__ == '__main__':
	df_emergencies = pd.read_csv(DatasetPaths.SAMUR)
	df_hospitals = pd.read_csv(DatasetPaths.HOSPITALS)
	df_medical_centers = pd.read_csv(DatasetPaths.MEDICAL_CENTERS)
	df_demographics = pd.read_csv(DatasetPaths.DEMOGRAPHICS)
	df_districts = pd.read_csv(DatasetPaths.DISTRICTS)
	df = merge_all(df_emergencies, df_hospitals, df_medical_centers, df_demographics, df_districts)
	print(df.shape)
	print(df.head())
	print(df.tail())
	df.to_csv(DatasetPaths.SAMUR_MERGED.format("all"))
	


