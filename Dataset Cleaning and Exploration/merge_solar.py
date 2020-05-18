import solarpy
import numpy as np
import pandas as pd
import pytz as tz
from datetime import datetime,timezone

import utils
import DatasetPaths

TIMEZONE_MADRID = tz.timezone("Europe/Madrid")
RADIAN_LIMIT = 0.10472

def get_sun_incidence_category(dt64):
	try:
		dt_zoned = pd.Timestamp(dt64).tz_localize(TIMEZONE_MADRID).to_pydatetime()
		dt_utc = dt_zoned.astimezone(tz = tz.UTC)
		azimuth = solarpy.solar_altitude(dt_utc,utils.KM0_LATITUDE)
		# Between -6º and +6º solar angle, it will be considerted twilight 
		if azimuth < -RADIAN_LIMIT: return "NIGHT"
		if azimuth >  RADIAN_LIMIT: return "DAY"
		if dt_utc.hour < 12:		return "SUNRISE"
		return "SUNSET"
	except:
		print('Ambiguous hour (due to save light time change: ', dt64)
		# All time changes are done at night:
		return 'NIGHT'
		
get_sun_incidence_category_vectorized = np.vectorize(get_sun_incidence_category)

def merge_solar(df_samur):
	df = df_samur.copy()
	df['Incidencia solar'] = get_sun_incidence_category_vectorized(df['Solicitud'])
	return df;

# Execute only if script run standalone (not imported)						
if __name__ == '__main__':
	df_samur = pd.read_csv(DatasetPaths.SAMUR, parse_dates=['Solicitud'])
	df = merge_solar(df_samur)
	print(df.head())
	df.to_csv(DatasetPaths.SAMUR_MERGED.format('solar'),index = False);
	
