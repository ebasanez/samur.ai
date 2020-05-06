import zipfile 
import os 
import sys

import pandas as pd

SEP = ';'

CSV = '.csv'
ZIP = '.zip'
PREFIX_FILENAME = 'pmed_ubicacion_'
FILENAME_SEPS = ['-', '_']

def update_with_interpolated_data_old(df, interpolated_data, idx, dates):
    small_data = df[df['id'] == idx]
    df_dates = small_data['fecha'].values
    for date in dates:
        if date not in df_dates:
            data = interpolated_data.loc[date]
            new_row = {'id': idx, 'fecha': date, 'carga': data['carga'], 'distrito': data['distrito']}
            df = df.append(new_row, ignore_index=True)
    return df

def update_with_interpolated_data(df, interpolated_data, district, dates):
    small_data = df[df['distrito'] == district]
    df_dates = small_data['fecha'].values
    for date in dates:
        if date not in df_dates:
            data = interpolated_data.loc[date]
            new_row = {'fecha': date, 'carga': data['carga'], 'distrito': data['distrito']}
            df = df.append(new_row, ignore_index=True)
    return df

def get_interpolated_data(grouped, dates):
    interpolated_data = pd.DataFrame(index=dates).join(grouped.set_index('fecha').resample('15min').asfreq().sort_index())
#    interpolated_data['id'] = interpolated_data['id'].fillna(method='ffill').fillna(method='bfill').astype(int)
    interpolated_data['distrito'] = interpolated_data['distrito'].fillna(method='ffill').fillna(method='bfill').astype(int)
    interpolated_data['carga'] = interpolated_data['carga'].interpolate(limit_direction='both')

    return interpolated_data

def prepare_historical_old(df):
    dates = df['fecha'].unique()
    num_dates = len(dates)
    # interpolate data for every date that doesn't exist
    grouped_by_id = df.groupby('id')
    for idx, g in grouped_by_id:
        if g.shape[0] != num_dates:
            interpolated_data = get_interpolated_data(g, dates)
            df = update_with_interpolated_data(df, interpolated_data, idx, dates)

    # once interpolated, we have all the needed data
    df = df.drop(labels='id', axis=1)
    df = df.groupby(['fecha', 'distrito']).mean().reset_index()

    return df

def prepare_historical(df):
    # remove not interesting columns
    df = df.drop(labels='id', axis=1)
    # group by fecha-distrito to obtain the average of carga
    df = df.groupby(['fecha', 'distrito']).mean().reset_index()
    # interpolate data for every date that doesn't exist on every district
    dates = df['fecha'].unique()
    num_dates = len(dates)
    grouped_by_district = df.groupby('distrito')
    for district, g in grouped_by_district:
        if g.shape[0] != num_dates:
            interpolated_data = get_interpolated_data(g, dates)
            df = update_with_interpolated_data(df, interpolated_data, district, dates)

    df['distrito'] = df['distrito'].astype(int)

    return df

def prepare_points(df):
    # get just the URB area (not M30)
    df = df[df['tipo_elem'] == 'URB']
    # fill the NaN id values
    df = df.sort_values(by='id').fillna(method='ffill')
    df = df.fillna(method='bfill')
    # convert the district to int
    df['distrito'] = df['distrito'].astype(int)

    return df

def get_points_csv(dir_points, month, year):
    for fsep in FILENAME_SEPS:
        name = '{}{}{}{}{}'.format(PREFIX_FILENAME, month, fsep, year, CSV)
        full_dir = os.path.join(dir_points, name)
        if os.path.exists(full_dir):
            return full_dir
    return ''

def main(dir_historical, dir_points):
    df_global = pd.DataFrame()
    for hist_zip in os.listdir(dir_historical):
        zipref = zipfile.ZipFile(os.path.join(dir_historical, hist_zip), 'r')
        name = hist_zip.split('.')[0]
        csv_historical = name + CSV
        unzipped = zipref.extract(csv_historical, path=dir_historical)
        if unzipped != os.path.join(dir_historical, csv_historical):
            print('An error has ocurred while unzipping {}'.format(os.path.join(dir_historical, csv_historical)))
            continue

        month, year = name.split('-')
        csv_points = get_points_csv(dir_points, month, year)
        if csv_points == '':
            print('An error has ocurred while opening points data from {}-{}'.format(month, year))
            os.remove(unzipped)
            continue

        df_historical = pd.read_csv(unzipped, sep=SEP, parse_dates=[1], usecols=['id', 'fecha', 'carga'])
        df_points = pd.read_csv(csv_points, sep=SEP, usecols=['id', 'distrito', 'tipo_elem'])

        df_points = prepare_points(df_points)
        merged_data = df_historical.merge(df_points[['id', 'distrito']], on='id')
        df_historical = prepare_historical(merged_data)

        df_global = pd.concat([df_global, df_historical], ignore_index=True)

        os.remove(unzipped)

    df_global.sort_values('fecha').to_csv('traffic_data.csv', sep=SEP, index=False)

def usage():
    print('Usage: {} <historical_dir> <points_dir>')
    print('where:')
    print('<historical_dir> is a directory with historical files in .zip')
    print('<points_dir> is a directory with points files (either in .csv or .zip)')
    sys.exit(-1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()

    dir_historical = sys.argv[1]
    dir_points = sys.argv[2]
    if not os.path.isdir(dir_historical) or not os.path.isdir(dir_points):
        usage()

    main(dir_historical, dir_points)