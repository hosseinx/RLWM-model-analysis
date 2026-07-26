import os
import csv


BASE_PATH = '/srv/black/data/rlwm'
DATA_FILE = os.path.join(BASE_PATH, 'dados', 'IED_Small_sample_N=14.csv')


with open(DATA_FILE, 'r') as datafile:

    datareader = csv.reader(datafile, delimiter=';')

    for row in datareader:
        print(row)

