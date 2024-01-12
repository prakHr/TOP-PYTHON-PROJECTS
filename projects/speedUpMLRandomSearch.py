# from sklearnex import patch_sklearn
# patch_sklearn()
import time
time.sleep(20)
from sklearn import tree
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.datasets import load_diabetes
def speedup_sklearn_model(x,y,sklearn_model,params,test_size=0.2,random_state=1234):
	
	start = timer()
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
	time_opt = timer() - start
	
	tuning_model = RandomizedSearchCV(sklearn_model,param_distributions = params).fit(x_train,y_train)
	best_model_params = tuning_model.best_params_
	model = tuning_model
	predicted = model.predict(x_test)
	
	print(f"Intel® extension for Scikit-learn time: {time_opt:.2f} s")
	return {
		"time":time_opt,
		"predicted":predicted,
		"x_train":x_train,
		"x_test":x_test,
		"y_train":y_train,
		"y_test":y_test,
		"best_model":model
	}

if __name__=="__main__":
	sklearn_model = tree.DecisionTreeRegressor()
	X, y = load_diabetes(return_X_y=True)

	params ={"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11,12],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
	
	rv = speedup_sklearn_model(X,y,sklearn_model,params,test_size=0.2,random_state=1234)
	print(rv)