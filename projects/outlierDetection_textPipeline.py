def outlier_detection_text_pipeline(df,text_column,label_column,test_size = 0.1,cv_n_folds = 5 ):
    import re
    import string
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sentence_transformers import SentenceTransformer

    from cleanlab.classification import CleanLearning
    
    def argCheck_outlier_detection_text_pipeline(df,text_column,label_column,test_size,cv_n_folds):
        if type(test_size)!=float:
            return f"{test_size} not a float value!"
        if 0.0<test_size<1.0:
            return f"{test_size} not in range (0,1)."
        if type(cv_n_folds)!=int:
            return f"{cv_n_folds} not a integer!"
        if cv_n_folds<=0:
            return f"{cv_n_folds} not > 0."
        cols = list(df.columns)
        Columns = [text_column,label_column]
        for Column in Columns:
            if Column not in cols:
                return f"{Column} not one of {cols}."
        return True
    check = argCheck_outlier_detection_text_pipeline(df,text_column,label_column,test_size,cv_n_folds)
    if not check:
        return check
    rv = {}
    raw_texts,raw_labels = df[text_column].values, df[label_column].values
    raw_train_texts, raw_test_texts, raw_train_labels, raw_test_labels = train_test_split(raw_texts, raw_labels, test_size=test_size)

    num_classes = len(set(raw_train_labels))

    print(f"This dataset has {num_classes} classes.")
    print(f"Classes: {set(raw_train_labels)}")

    encoder = LabelEncoder()
    encoder.fit(raw_train_labels)

    train_labels = encoder.transform(raw_train_labels)
    test_labels = encoder.transform(raw_test_labels)

    transformer = SentenceTransformer('google/electra-small-discriminator')

    train_texts = transformer.encode(raw_train_texts)
    test_texts = transformer.encode(raw_test_texts)

    model = LogisticRegression(max_iter=400)


    # for efficiency; values like 5 or 10 will generally work better

    cl = CleanLearning(model, cv_n_folds=cv_n_folds)

    label_issues = cl.find_label_issues(X=train_texts, labels=train_labels)
    label_issues_df = pd.DataFrame(
        {
            "text": raw_train_texts,
            "given_label": raw_train_labels,
            "predicted_label":encoder.inverse_transform(label_issues["predicted_label"])
        }
    )
    rv["label issues"] = label_issues_df 
    # rv["label issues"] = label_issues_df.to_dict()
    identified_issues = label_issues[label_issues["is_label_issue"] == True]
    print(f"cleanlab found {len(identified_issues)} potential label errors in the dataset.")
    baseline_model = LogisticRegression(max_iter=400)  # note we first re-instantiate the model
    baseline_model.fit(X=train_texts, y=train_labels)
    
    preds = baseline_model.predict(test_texts)
    acc_og = accuracy_score(test_labels, preds)
    rv["original accuracy"] = acc_og
    print(f"\n Test accuracy of original model: {acc_og}")
    cl.fit(X=train_texts, labels=train_labels, label_issues=cl.get_label_issues())
    pred_labels = cl.predict(test_texts)
    acc_cl = accuracy_score(test_labels, pred_labels)
    print(f"Test accuracy of cleanlab's model: {acc_cl}")
    rv["improved accuracy"] = acc_cl
    return rv







if __name__=="__main__":
    import pandas as pd
    path = r"C:\Users\gprak\Downloads\projects\Data\banking-intent-classification.csv"
    df = pd.read_csv(path)
    text_column = "text"
    label_column = "label"
    rv = outlier_detection_text_pipeline(df,text_column,label_column)
    from pprint import pprint
    pprint(rv)

  

