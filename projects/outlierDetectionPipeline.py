from outlierDetection_textPipeline import outlier_detection_text_pipeline
from outlierDetection_tabularPipeline import outlier_detection_tabular_pipeline
def outlierDetection_multimodalPipeline(config_dict,type):
	rv = -1
	if type == "text":
		rv = outlier_detection_text_pipeline(pd.read_csv(config_dict["path"]),config_dict["text_column"],config_dict["label_column"])
	elif type == "tabular":
		rv = outlier_detection_tabular_pipeline(config_dict["X"],config_dict["labels"],config_dict["clf"],config_dict["numeric_features"])
	return rv

if __name__=="__main__":
	import pandas as pd
	path = r"C:\Users\gprak\Downloads\projects\Data\banking-intent-classification.csv"
	config_dict = {
	    "path" : path,
	    "text_column" : "text",
	    "label_column" : "label"
	    
	}
	type = "text"
	rv = outlierDetection_multimodalPipeline(config_dict,type)
	from pprint import pprint
	pprint(rv)

	import pandas as pd
	from sklearn.preprocessing import StandardScaler, LabelEncoder
	from sklearn.ensemble import ExtraTreesClassifier

	path = r"C:\Users\gprak\Downloads\projects\Data\grades-tabular-demo-v2.csv"
	grades_data = pd.read_csv(path)
	X_raw = grades_data[["exam_1", "exam_2", "exam_3", "notes"]]
	labels_raw = grades_data["letter_grade"]
	categorical_features = ["notes"]
	X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)

	numeric_features = ["exam_1", "exam_2", "exam_3"]
	scaler = StandardScaler()
	X_processed = X_encoded.copy()
	X_processed[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])
	encoder = LabelEncoder()
	encoder.fit(labels_raw)
	labels = encoder.transform(labels_raw)
	clf = ExtraTreesClassifier()
	    # rv = outlier_detection_tabular_pipeline(X_processed,labels,clf,numeric_features)config_dict = {
	config_dict ={
			"X":X_processed,
			"labels":labels,
			"clf":clf,
			"numeric_features":numeric_features
	}
	type = "tabular"
	rv = outlierDetection_multimodalPipeline(config_dict,type)
	pprint(rv)