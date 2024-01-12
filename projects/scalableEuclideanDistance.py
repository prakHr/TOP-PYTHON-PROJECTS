import Orange
import numpy as np
def scalable_euclidean_distance(dataset : Orange.data.Table,index : int) -> np.ndarray:
	eucl_dist = Orange.distance.Euclidean(dataset[index],dataset)
	return eucl_dist

def scalable_euclidean_distances(dataset : Orange.data.Table) -> dict:
	x = len(dataset)
	rv = {}
	import pandas as pd
	import numpy as np
	df = pd.DataFrame()
	df["Euclidean Distance"] = ["scalable_euclidean_distance"+"(dataset,"+str(i)+")" for i in range(x)]
	df["output"] = df["Euclidean Distance"].apply(lambda x:eval(x))
	return df.to_dict()
	

if __name__=="__main__":
	csv_path = r"C:\Users\gprak\Downloads\projects\Data\1000000 Sales Records\1000000 Sales Records.csv"
	dataset = Orange.data.Table(csv_path)
	dataset = dataset
	print(len(dataset))
	rv = scalable_euclidean_distances(dataset)
	from pprint import pprint
	pprint(rv)
