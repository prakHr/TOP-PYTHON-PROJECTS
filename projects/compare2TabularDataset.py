import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from pycm import *
import seaborn as sns
import matplotlib.pyplot as plt

def compare(df1,df2,feature_columns,class_name_column,sklearn_model,show_plt = True):
	X1         = df1[feature_columns].values.tolist()
	y_actu1    = df1[class_name_column].values.tolist()
	clf1       = sklearn_model
	clf1.fit(X1,y_actu1)
	y_pred1    = clf1.predict(X1)
	cm1        = ConfusionMatrix(actual_vector=y_actu1, predict_vector=y_pred1)
	np_array1  = cm1.to_array(normalized=True)

	X2         = df2[feature_columns].values.tolist()
	y_actu2    = df2[class_name_column].values.tolist()
	clf2       = sklearn_model
	clf2.fit(X2,y_actu2)
	y_pred2    = clf2.predict(X2)
	cm2        = ConfusionMatrix(actual_vector=y_actu2, predict_vector=y_pred2)
	np_array2  = cm2.to_array(normalized=True)
	samples    = [np_array1,np_array2]
	cf_matrix  = {"cm1":samples[0],"cm2":samples[1]}
	fig, axn = plt.subplots(1,2, sharex=True, sharey=True,figsize=(12,2))

	for i, ax in enumerate(axn.flat):
	    k = list(cf_matrix)[i]
	    sns.heatmap(cf_matrix[k], ax=ax,cbar=i==4)
	    ax.set_title(k,fontsize=8)
	# ax         = sns.violinplot(data = samples)
	# sns.displot(samples[0])
	# sns.displot(samples[1])
	# cm1.plot(cmap=plt.cm.Greens,normalized = True, number_label=True, plot_lib="matplotlib")
	# cm2.plot(cmap=plt.cm.Greens,normalized = True, number_label=True, plot_lib="matplotlib")
	if show_plt == True:
		plt.show()
	return {
		"cm1":np_array1,
		"cm2":np_array2
	}

if __name__=="__main__":
	path = r"C:\Users\gprak\Downloads\projects\Data\archive\Iris.csv"
	df1 = pd.read_csv(path)
	df2 = pd.read_csv(path)
	feature_columns = ['SepalLengthCm','SepalWidthCm']
	class_name_column = 'Species'
	sklearn_model = RandomForestClassifier()
	rv = compare(df1,df2,feature_columns,class_name_column,sklearn_model,show_plt = False)
	from pprint import pprint
	pprint(rv)

