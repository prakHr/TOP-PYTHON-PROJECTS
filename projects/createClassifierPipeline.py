import milk
import numpy as np
import pandas as pd

def supervised_clf_pipeline(features,labels,clf=milk.supervised.defaultclassifier()):
    model = clf.train(features, labels)
    return {"supervised_clf":model}

def update_supervised_clf_with_new_dataset(clf,updated_features):
    new_labels = clf.apply(updated_features)
    return {"new_labels" : new_labels , "supervised_clf" : clf}

def unsupervised_clf_pipeline(features, algo_name = "milk.unsupervised.pca"):
    rv =algo_name+"(features)"
    rv = eval(rv)
    return rv




if __name__=="__main__":
    # train the ml pipeline
    features = np.random.randn(100,20)
    features[:50] *= 2
    labels = np.repeat((0,1), 50)
    clf = milk.supervised.defaultclassifier()
    rv = supervised_clf_pipeline(features,labels,clf)
    model = rv["supervised_clf"]

    # update the classifier on new dataset(idea we can use realtime dataset to update)
    rv = update_supervised_clf_with_new_dataset(model,np.random.randn(20))
    new_labels = rv["new_labels"]
    model = rv["supervised_clf"]


    # train the ml pipeline
    features = np.random.randn(100,20)
    features[:50] *= 2
    rv = unsupervised_clf_pipeline(features,"milk.unsupervised.center")
    print(rv)