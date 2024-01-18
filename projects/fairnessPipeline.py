import pandas as pd
from fairlearn.datasets import fetch_boston
import fairlearn.metrics as fm 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing
def fairness_featurewise_pipeline(df,model,feature_col,features_col,class_col):
	X = df[features_col]
	y = df[class_col]
	X_train,X_test,y_train,y_test = train_test_split(X,y)
	model.fit(X_train,y_train)
	predicted = model.predict(X_test)

	rv = fm.demographic_parity_difference(
			y_true=y_test,
			y_pred=predicted,
			sensitive_features = X_test[feature_col]
		)
	
	return rv

def argCheck(df,model,class_col,features_col):
	if type(df)!=pd.DataFrame:
		return f"please provide a pandas dataframe object."
	cols = list(df.columns)
	if type(features_col)!=list:
		return f"please provide a list of features."
	if class_col not in cols:
		return f"{class_col} not present in {cols}"
	for feature_col in features_col:
		if feature_col not in cols:
			return f"{feature_col} not present in {cols}"
	return True
def fairness_pipeline(df,model,class_col,features_col):
	check = argCheck(df,model,class_col,features_col)
	if not check:return check
	rv = {}
	for i,feature_col in enumerate(features_col):
		rv[f"{feature_col}"] = fairness_featurewise_pipeline(df,model,feature_col,features_col,class_col)
	return rv
if __name__=="__main__":
	# X,y = fetch_boston(return_X_y = True)
	# print(type(X))
	path = r"C:\Users\gprak\Downloads\projects\Data\archive\Iris.csv"
	df = pd.read_csv(path)

	features_col = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
	class_col = 'Species'
	le = preprocessing.LabelEncoder()
	df[class_col] = le.fit_transform(df[class_col].values)
	# print(df.head(20))
	model = LogisticRegression(random_state=123, solver = 'liblinear')
	rv = fairness_pipeline(df,model,class_col,features_col)
	from pprint import pprint
	pprint(rv)