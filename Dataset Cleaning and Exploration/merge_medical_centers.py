import sys
from datetime import datetime
import ast
import numpy as np
import pandas as pd
import DatasetPaths

'''
Expands SAMUR emergencies dataset by adding the number of medical centers open in the same district at the moment of the emergency communication
'''
class CentroMedicoOpenChecker:

	def __init__(self, df_centros_medicos):
		self.df = df_centros_medicos
		self._is_open_vectorized = np.vectorize(self._is_open)
		
	def getNumCentrosMedicosOpen(self, district, datetimeStr):
		dt = datetime.strptime(datetimeStr,'%Y-%m-%d %H:%M:%S');
		#d = self.df[self.df['DISTRITO'] == district]
		d = self.df[(self.df['DISTRITO'] == district) & (self._is_open_vectorized(self.df['TIME'],dt))]
		#d = self.df[self._is_open(self.df.TIME,dt)]
		self.df.DISTRITO = self.df.DISTRITO.astype(str)
		return d.shape[0]

	@staticmethod	
	def _is_open(calendar, datetime):
		month = datetime.month
		day = datetime.day
		weekday = datetime.weekday()
		hour = datetime.hour
		minute = datetime.minute
		for d in calendar:
			#Check month and day
			try:
				if (month > d['M_INIT'] or month == d['M_INIT'] & day >= d['D_INIT']) & (month < d['M_END'] or month == d['M_END'] & day <= d['D_END']):
					tt = d['TT'][weekday]
					for t in tt:
						if(hour > t[0] or hour == t[0] & minute >= t[1]) & (hour < t[2] or hour == t[2] & minute <= t[3]):
							return True
			except Exception as e: 
				print(f"EXCEPT: {e} {datetime},{d}")
				print(f"EXCEPT: mes {month}, day {day}, weekday {weekday}, hour {hour}, minute {minute}, lapse {d}")
				print(f"EXCEPT: mes {d['M_INIT']}, day {d['D_INIT']}")
		return False;
	  
def merge_medical_centers(ds_emergencies, ds_medical_centers):
	ds_medical_centers['TIME'] = ds_medical_centers['TIME'].apply(ast.literal_eval)
	openChecker = CentroMedicoOpenChecker(ds_medical_centers)
	getNumCentrosMedicosOpen = np.vectorize(openChecker.getNumCentrosMedicosOpen)
	ds_emergencies['centros_medicos_open'] = getNumCentrosMedicosOpen(ds_emergencies['Distrito'],ds_emergencies['Solicitud'])
	return ds_emergencies;
	

if __name__ == '__main__':	
	ds = pd.read_csv(DatasetPaths.SAMUR)
	dsc = pd.read_csv(DatasetPaths.MEDICAL_CENTERS, dtype = {'TIME':object})	  
	ds = merge_medical_centers(ds,dsc)
	ds.to_csv('datasets/SAMUR_medical_centers.csv', index = False)