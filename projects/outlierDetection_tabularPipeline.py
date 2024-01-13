def outlier_detection_tabular_pipeline(X,labels,clf,numeric_features,num_crossval_folds = 5,test_size = 0.2):
    import random
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import ExtraTreesClassifier

    from cleanlab.filter import find_label_issues
    from cleanlab.classification import CleanLearning

    SEED = 1234

    np.random.seed(SEED)
    random.seed(SEED)
    def argCheck_outlier_detection_tabular_pipeline(X,labels,clf,numeric_features,num_crossval_folds,test_size):
        if type(X)!=np.ndarray:
            return "x is not of type np.ndarray!"
        if type(labels)!=np.ndarray:
            return "labels is not of type np.ndarray!"
        if type(numeric_features)!=list:
            return "numeric_features is not of type list!"
        if type(num_crossval_folds)!=int and num_crossval_folds<=0:
            return "either num_crossval_folds is not integer or num_crossval_folds is <=0!"
        if type(test_size)!=float and not 0.0<test_size<1.0:
            return "either test_size is not float or test_size is not b/w (0,1)"    
        return True

    check = argCheck_outlier_detection_tabular_pipeline(X,labels,clf,numeric_features,num_crossval_folds,test_size)
    if not check:
        return check
    pred_probs = cross_val_predict(
        clf,
        X,
        labels,
        cv=num_crossval_folds,
        method="predict_proba",
    )
    rv = {}
    ranked_label_issues = find_label_issues(
        labels=labels, pred_probs=pred_probs, return_indices_ranked_by="self_confidence"
    )
    rv["ranked label issues"] = ranked_label_issues

    print(f"Cleanlab found {len(ranked_label_issues)} potential label errors.")
    X_train, X_test, labels_train, labels_test = train_test_split(
        X,
        labels,
        test_size=test_size,
        random_state=SEED,
    )
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
    X_test[numeric_features] = scaler.transform(X_test[numeric_features])
    
    clf.fit(X_train, labels_train)
    acc_og = clf.score(X_test, labels_test)
    print(f"Test accuracy of original model: {acc_og}")
    rv["original accuracy"] = acc_og
    clf = ExtraTreesClassifier()  # Note we first re-initialize clf
    cl = CleanLearning(clf)  # cl has same methods/attributes as clf
    _ = cl.fit(X_train, labels_train)
    preds = cl.predict(X_test)
    acc_cl = accuracy_score(labels_test, preds)
    print(f"Test accuracy of cleanlab-trained model: {acc_cl}")
    rv["improved accuracy"] = acc_cl

    return rv

if __name__=="__main__":
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
    rv = outlier_detection_tabular_pipeline(X_processed,labels,clf,numeric_features)
    from pprint import pprint
    pprint(rv)

