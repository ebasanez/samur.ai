import pandas as pd
import DatasetPaths
from utils import sumCells
from statistics import mean 

kpis = ['density','surface_ha','age_avg','foreigns_perc','people_per_home','elder_alone_perc','monoparental_homes_perc','income','unemployment_perc']

cells2016 = {
			'density':[(3,3)],
			'surface_ha':[(2,3)],
			'age_avg':[(7,4)],
			'foreigns_perc':[(21,3)],
			'people_per_home':[(27,4)],
			'elder_alone_perc':[(29,3),(30,3)],
			'monoparental_homes_perc':[(31,3),(32,3)],
			'income':[(42,4)],
			'unemployment_perc':[(51,4)]
			}
cells2017 = {
			'density':[(3,3)],
			'surface_ha':[(2,3)],
			'age_avg':[(9,4)],
			'foreigns_perc':[(25,3)],
			'people_per_home':[(27,4)],
			'elder_alone_perc':[(34,3),(35,3)],
			'monoparental_homes_perc':[(36,3),(37,3)],
			'income':[(49,4)],
			'unemployment_perc':[(67,3)]
			
			}
cells2018 = {
			'density':[(3,3)],
			'surface_ha':[(2,3)],
			'age_avg':[(9,4)],
			'foreigns_perc':[(28,3)],
			'people_per_home':[(36,4)],
			'elder_alone_perc':[(37,3),(38,3)],
			'monoparental_homes_perc':[(39,3),(40,3)],
			'income':[(55,4)],
			'unemployment_perc':[(67,3)]
			}

def getDemographicsDataset(df_districts):

	xls2016 = pd.ExcelFile(DatasetPaths.RAW_DEMOGRAPHICS_2016)
	xls2017 = pd.ExcelFile(DatasetPaths.RAW_DEMOGRAPHICS_2017)
	xls2018 = pd.ExcelFile(DatasetPaths.RAW_DEMOGRAPHICS_2018)

	districts = xls2016.sheet_names
	df = buildBaseDataframe(districts)
	
	## Each dataset have its own cell positions for items:

	for district in districts:
		ds2016 = pd.read_excel(xls2016,district)
		ds2017 = pd.read_excel(xls2017,district)
		ds2018 = pd.read_excel(xls2018,district)
		for kpi in kpis:
			value = mean([sumCells(ds2016,cells2016[kpi]),sumCells(ds2017,cells2017[kpi]),sumCells(ds2018,cells2018[kpi])])
			df.loc[df['District']==district,kpi] = value
	#Replace district name with normalized one		
	df_district_names = df_districts[['DATASET_SAMUR','DATASET_DEMOGRAPHICS']]
	df = df.merge(df_district_names,left_on='District', right_on='DATASET_DEMOGRAPHICS')		
	df['District'] = df['DATASET_SAMUR']
	df.drop(columns=['DATASET_SAMUR','DATASET_DEMOGRAPHICS'],inplace = True)
	return df
		
def buildBaseDataframe(districts):
	return pd.DataFrame(districts, columns = ['District'])


if __name__ == '__main__':	
	df_districts = pd.read_csv(DatasetPaths.DISTRICTS, encoding = 'utf8')
	df = getDemographicsDataset(df_districts)	
	print(df.head())
	df.to_csv(DatasetPaths.DEMOGRAPHICS,index=False)