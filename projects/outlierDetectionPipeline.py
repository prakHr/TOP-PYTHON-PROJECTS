from outlierDetection_textPipeline import outlier_detection_text_pipeline
from outlierDetection_tabularPipeline import outlier_detection_tabular_pipeline
from outlierDetection_audioPipeline import outlier_detection_audio_pipeline
from outlierDetection_imagePipeline import outlier_detection_image_pipeline

def outlierDetection_multimodalPipeline(config_dict,type):
	rv = -1
	if type == "text":
		rv = outlier_detection_text_pipeline(pd.read_csv(config_dict["path"]),config_dict["text_column"],config_dict["label_column"])
	elif type == "tabular":
		rv = outlier_detection_tabular_pipeline(config_dict["X"],config_dict["labels"],config_dict["clf"],config_dict["numeric_features"])
	elif type == "audio":
		rv = outlier_detection_audio_pipeline(pd.read_csv(config_dict["path"]),config_dict["audio_column"],config_dict["label_column"])
	elif type == "image":
		rv = outlier_detection_image_pipeline(pd.read_csv(config_dict["dataset"]),config_dict["label_column"],config_dict["image_column"])
	return rv

if __name__=="__main__":
	from datasets import load_dataset
    dataset = load_dataset("fashion_mnist", split="train")
    label_column = "label"
    image_column = "image"
    config_dict = {
    	"dataset":dataset,
    	"label_column":label_column,
    	"image_column":image_column
    }
    type = "image"
    rv = outlierDetection_multimodalPipeline(config_dict,type)
    from pprint import pprint
    pprint(rv)
	#######################################################################################
	# import pandas as pd
	# path = r"C:\Users\gprak\Downloads\projects\Data\banking-intent-classification.csv"
	# config_dict = {
	    # "path" : path,
	    # "text_column" : "text",
	    # "label_column" : "label"
	    
	# }
	# type = "text"
	# rv = outlierDetection_multimodalPipeline(config_dict,type)
	# from pprint import pprint
	# pprint(rv)

	#######################################################################################
	# import pandas as pd
	# from sklearn.preprocessing import StandardScaler, LabelEncoder
	# from sklearn.ensemble import ExtraTreesClassifier

	# path = r"C:\Users\gprak\Downloads\projects\Data\grades-tabular-demo-v2.csv"
	# grades_data = pd.read_csv(path)
	# X_raw = grades_data[["exam_1", "exam_2", "exam_3", "notes"]]
	# labels_raw = grades_data["letter_grade"]
	# categorical_features = ["notes"]
	# X_encoded = pd.get_dummies(X_raw, columns=categorical_features, drop_first=True)

	# numeric_features = ["exam_1", "exam_2", "exam_3"]
	# scaler = StandardScaler()
	# X_processed = X_encoded.copy()
	# X_processed[numeric_features] = scaler.fit_transform(X_encoded[numeric_features])
	# encoder = LabelEncoder()
	# encoder.fit(labels_raw)
	# labels = encoder.transform(labels_raw)
	# clf = ExtraTreesClassifier()
	#     # rv = outlier_detection_tabular_pipeline(X_processed,labels,clf,numeric_features)config_dict = {
	# config_dict ={
	# 		"X":X_processed,
	# 		"labels":labels,
	# 		"clf":clf,
	# 		"numeric_features":numeric_features
	# }
	# type = "tabular"
	# rv = outlierDetection_multimodalPipeline(config_dict,type)
	# pprint(rv)

	#############################################################################################
	# path = r"wav_path.csv"
    # audio_column = "wav_audio_file_path"
    # label_column = "label"
    # config_dict = {
    	# "path":pd.read_csv(path),
    	# "audio_column":audio_column,
    	# "label_column":label_column
    # }
    # type = "audio"
    # rv = outlierDetection_multimodalPipeline(config_dict,type)
    # pprint(rv)