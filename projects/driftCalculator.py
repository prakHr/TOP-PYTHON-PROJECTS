
def drift_calculation(type,train_dataset,test_dataset):
	from deepchecks.tabular.checks import FeatureDrift, MultivariateDrift,LabelDrift,PredictionDrift
	from deepchecks.nlp.checks import TextEmbeddingsDrift,PropertyDrift,LabelDrift as nlpLabelDrift,PredictionDrift as nlpPredictionDrift
	from deepchecks.vision.checks import ImagePropertyDrift,ImageDatasetDrift,LabelDrift as ImageLabelDrift,PredictionDrift as ImagePredictionDrift

	rv = -1
		
	if type == "tabular":
		tabular_drifts = [FeatureDrift(),MultivariateDrift(),LabelDrift()]
		rv = [drift.run(train_dataset = train_dataset,test_dataset = test_dataset) for drift in tabular_drifts]
	elif type == "nlp":
		nlp_drifts = [nlpLabelDrift()]
		rv = [drift.run(train_dataset = train_dataset,test_dataset = test_dataset) for drift in nlp_drifts]
	elif type == "vision":	
		vision_drifts = [ImagePropertyDrift(),ImageDatasetDrift(),ImageLabelDrift()]
		rv = [drift.run(train_dataset = train_dataset,test_dataset = test_dataset) for drift in vision_drifts]

	return rv

if __name__=="__main__":
	from sklearn.model_selection import train_test_split
	import pandas as pd
	import numpy as np
	from deepchecks.tabular import Dataset
	from deepchecks.nlp import TextData
	from deepchecks.vision.checks import ImagePropertyDrift
	from deepchecks.vision.datasets.detection import coco_torch as coco
	from deepchecks.vision.utils import image_properties
	from pprint import pprint

	train_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=[1,0], p=[0.5, 0.5], size=(1000, 1))], axis=1)
	test_data = np.concatenate([np.random.randn(1000,2), np.random.choice(a=[1,0], p=[0.35, 0.65], size=(1000, 1))], axis=1)

	df_train = pd.DataFrame(train_data, columns=['col1', 'col2', 'target'])
	df_test = pd.DataFrame(test_data, columns=['col1', 'col2', 'target'])

	train_dataset = Dataset(df_train, label='target')
	test_dataset = Dataset(df_test, label='target')
	tabular_drift = drift_calculation("tabular",train_dataset,test_dataset)
	pprint(tabular_drift)


	from deepchecks.nlp.datasets.classification import tweet_emotion
	train, test = tweet_emotion.load_data(data_format='DataFrame')
	train = TextData(train.text, label=train['label'], task_type='text_classification',
                 metadata=train.drop(columns=['label', 'text']))
	test = TextData(test.text, label=test['label'], task_type='text_classification',
                metadata=test.drop(columns=['label', 'text']))
	nlp_drift = drift_calculation("nlp",train,test)
	pprint(nlp_drift)


	from deepchecks.vision.utils import image_properties
	train_dataset = coco.load_dataset(train=True, object_type='VisionData')
	test_dataset = coco.load_dataset(train=False, object_type='VisionData')
	cv_drift = drift_calculation("vision",train_dataset,test_dataset)
	pprint(cv_drift)