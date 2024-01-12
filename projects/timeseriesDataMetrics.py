from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_f1_score_timeseries(target_times,readings,test_size=0.25,random_state=0,pipeline_name='lstm_with_unstack',metrics = 'f1_score'):
	from draco import get_pipelines
	from draco.pipeline import DracoPipeline
	train,test=train_test_split(target_times,test_size= test_size,random_state = random_state)
	test_targets = test.pop('target')
	pipeline  = DracoPipeline(pipeline_name)
	pipeline.fit(train,readings)
	predictions = pipeline.predict(test,readings)
	score = eval(f"{metrics}(test_targets,predictions)")
	return score


def get_serial_f1_score_timeseries(target_times,readings,pipelines_name,test_size=0.25,random_state=0,metrics = 'f1_score'):
	from draco import get_pipelines
	from draco.pipeline import DracoPipeline
	train,test=train_test_split(target_times,test_size= test_size,random_state = random_state)
	test_targets = test.pop('target')
	pipelines = [DracoPipeline(pipeline_name) for pipeline_name in pipelines_name]
	
	for pipeline in pipelines:
		pipeline.fit(train,readings)
		predictions = pipeline.predict(test,readings)

	score = eval(f"{metrics}(test_targets,predictions)")
	return score

if __name__=="__main__":
	from draco.demo import load_demo
	target_times,readings = load_demo()
	score = get_f1_score_timeseries(target_times,readings,metrics='accuracy_score',pipeline_name='double_lstm_with_unstack')
	print(score)

	score = get_serial_f1_score_timeseries(target_times,readings,metrics='accuracy_score',pipelines_name=['lstm_with_unstack','double_lstm_with_unstack'])
	print(score)