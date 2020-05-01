import pandas as pd
from utils import *

# Calculates x and y coordenates in km, relative to Madrid Km0 point
def preproces_hospitals(df):
	df['hospital_x'] = longitudeToKm0(df.longitude)
	df['hospital_y'] = latitudeToKm0(df.latitude)
	return df

if __name__ == '__main__':	
	df_hospitals = pd.read_csv(DatasetPaths.RAW_HOSPITALS, encoding = 'utf8')
	df = preproces_hospitals(df_hospitals)	
	print(df.head())
	df.to_csv(DatasetPaths.HOSPITALS, index=False)