import sys, getopt
import DatasetPaths
import numpy as np
from decimal import Decimal

LATITUDE_1DEG_KM = 110.574
LONGITUDE_KM_PER_DEG = 111.320
KM0_LATITUDE = 40.4146500
KM0_LONGITUDE = -3.7004000	

def append_suffix(filename, suffix):
    return "{0}_{2}.{1}".format(*filename.rsplit('.', 1) + [suffix])

def getDSOtherColumnNames(df, exclusion):
	columns = list(df.columns.values)
	for e in exclusion:
		columns.remove(e)
	return columns
	
def obtainFilenamesFromOptions(feature_file, suffix, samur_file=DatasetPaths.SAMUR):
	try:
		opts, args = getopt.getopt(sys.argv[1:],"mhs:f:o:",["samurfile=","featurefile=","outputfile"])
		print('Options: ',opts)
	except getopt.GetoptError:
		print('merge_[feature].py -s <samurfile> -f <faturefile> -o <outputfile> -m')
		sys.exit(2)
	
	# Default values:	
	minify = False	
	output_file = append_suffix(samur_file, suffix)  
	for opt, arg in opts:
		if opt == '-h':
			print('merge_[feature].py -s <samurfile> -f <faturefile> -o <outputfile>')
			sys.exit()
		elif opt in ("-s", "--samurfile"):
			samur_file = arg
		elif opt in ("-f", "--featurefile"):
			feature_file = arg
		elif opt in ("-o", "--outputfile"):
			output_file = arg
		elif opt in ("-m", "--minify"):
			minify = True
	print(f'SAMUR file: {samur_file}')
	print(f'Feature file: {feature_file}')
	print(f'Output file: {output_file}')
	print(f'Minify: {minify}\n')
	return samur_file, feature_file, output_file, minify
	
def sumCells(df, cells):
	cells = [parseNumberCell(df.iloc[cell[0],cell[1]]) for cell in cells]
	return sum(cells)
	
def parseNumberCell(cell):
	if type(cell) == str:
		return Decimal(cell.replace('.','').replace('%','').replace(',','.').replace(' ','')) 
	return Decimal(cell)

# TODO Pasar a radianes para el calculo del coseno	
def coordinatesToKm0(latitude, longitude):		
	x = np.round(np.cos(KM0_LATITUDE/180 * np.pi) * (longitude - KM0_LONGITUDE) * LONGITUDE_KM_PER_DEG, 9)
	y = np.round((latitude - KM0_LATITUDE) * LATITUDE_1DEG_KM, 9)
	return (x,y)
	
	
KM0_LATITUDE = 40.4146500 / 180*np.pi
KM0_LONGITUDE = -3.7004000 / 180*np.pi
KM0_RADIUS = 6371.7

def longitudeToKm0(longitude):
	lon_rad = longitude /180*np.pi
	return np.round((lon_rad - KM0_LONGITUDE) * KM0_RADIUS * np.cos(KM0_LATITUDE), 6)
	
def latitudeToKm0(latitude):
	lat_rad = latitude /180*np.pi
	return np.round((lat_rad - KM0_LATITUDE) * KM0_RADIUS, 6) 

def coordinatesToKm0(latitude, longitude):
	x = longitudeToKm0(longitude)
	y = latitudeToKm0(latitude) 
	return (x,y)