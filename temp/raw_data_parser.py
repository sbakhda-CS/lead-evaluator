import pandas as pd
from datetime import datetime
import time, requests, urllib, json


def col_to_list(sheet, col, b=2, e=-1):

    sheet = pd.read_excel(file, sheet_name=sheet)

    return sheet[col].values.tolist()[b:e]


# Result format: { i : (e, l), ... } where i is ID, e is earliest timestamp, l is latest timestamp
#  all timestamps are in Unix format
def get_times(sheet='Time'):

    earliest_times = [time.mktime(x.timetuple()) for x in col_to_list(sheet, 'Earliest')]
    latest_times = [time.mktime(x.timetuple()) for x in col_to_list(sheet, 'Latest')]

    return {i : (earliest_times[i], latest_times[i]) for i in range(0, len(earliest_times))}


# Result format: { i : r , ... } where i is ID, r is the json formatted google geocode API response
def get_address(sheet='Contact Details', key='AIzaSyDtrNE8DuzXlbClAc0dH6xgmOA_Jlsay8M'):

    all_address = [
        col_to_list(sheet, 'Company ID'),
        col_to_list(sheet, 'Street Address'),
        col_to_list(sheet, 'City'),
        col_to_list(sheet, 'State/Region'),
        col_to_list(sheet, 'Postal Code'),
        col_to_list(sheet, 'Country')
        ]

    results = dict()

    # ! only limited to 10 for now so that I don't use up my free API quota
    # for i in range(2, len(all_address[0]) - 1):
    for i in range(0, 10):
        address = []
        for j in range(1, len(all_address)):
            x = str(all_address[j][i])
            if x != 'nan':
                address.append(x)

        address = ', '.join(address)

        if address != '':
            res = json.loads(requests.request('GET', 'https://maps.googleapis.com/maps/api/geocode/json?address='
                             + urllib.parse.quote_plus(address)
                             + 'key='+key).content)
        else:
            res = {}

        results[all_address[0][i]] = res

    return results


if __name__  == "__main__":

    file = '/Users/sbakhda/Google Drive/Project Drive /HubSpot Data Exported 2/Companies & Contacts (Master Lists)/companies.xlsx'

    all_times = get_times(sheet='Time')
    all_addresses = get_address(sheet='Contact Details')
