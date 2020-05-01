import pandas as pd
import DatasetPaths
from utils import *

def expandMedicalCenters(df_centers, df_time):
	ds = pd.merge(df_centers, df_time, left_index=True, right_index=True)

	ds.drop(columns= ['DESCRIPCION-ENTIDAD','HORARIO','EQUIPAMIENTO','TIPO','DESCRIPCION','TRANSPORTE','ACCESIBILIDAD','CONTENT-URL','NOMBRE-VIA','CLASE-VIAL','TIPO-NUM','NUM','PLANTA','PUERTA','ESCALERAS','ORIENTACION','LOCALIDAD','PROVINCIA','BARRIO','CODIGO-POSTAL','COORDENADA-X','COORDENADA-Y','TELEFONO','FAX','EMAIL'],inplace=True)
	#TODO cambiar por la funcion de utils
	ds['x_km0'] = longitudeToKm0(ds.LONGITUD)
	ds['y_km0'] = latitudeToKm0(ds.LATITUD)
	pd.set_option('display.max_colwidth',100)
	#Remove Unnamed cols
	ds = ds.loc[:, ~ds.columns.str.contains('^Unnamed')]
	return ds;
	

if __name__ == '__main__':	
	df_centers = pd.read_csv(DatasetPaths.RAW_MEDICAL_CENTERS, sep = ';')
	print(df_centers.head())
	df_time = pd.read_csv(DatasetPaths.MEDICAL_CENTERS_TIMES)
	print(df_time.head())
	
	result = expandMedicalCenters(df_centers,df_time)
	result.to_csv(DatasetPaths.MEDICAL_CENTERS)
