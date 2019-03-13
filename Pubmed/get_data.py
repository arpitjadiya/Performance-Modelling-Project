import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_folder = "./PubMed_20k_RCT"

os.chdir(data_folder)

file_path = ["train.txt","test.txt"]

for file1 in file_path:
	x = []
	y = []
	print("reading file:", file1)
	with open(file1) as f:
    		data = f.readlines()

	data1 = []
	for d in data:
		data1.append(d.split())	

	for d in data1:
		if len(d) == 0:
			continue
		if d[0].lower() == 'background' or d[0].lower() == 'objective' or d[0].lower() == 'methods' or d[0].lower() == 'results' or d[0].lower() == 'conclusions':
			y.append(d[0])
			x.append(' '.join(d[1:]))

	print(len(y),len(x))
   
	data = {'news': x, 'type': y}       
	df = pd.DataFrame(data)
	print('writing csv file ...')
	if f == "train.txt":
		df.to_csv('../dataset.csv', index=False)
	else:
		df.to_csv('../dataset_test.csv', index=False)
