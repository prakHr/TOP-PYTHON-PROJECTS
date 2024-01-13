def outlier_detection_audio_pipeline(df,audio_column,label_column,test_size = 0.1,cv_n_folds = 5 ):
    import os
    import pandas as pd
    import numpy as np
    import random
    import tensorflow as tf
    import torch
    from cleanlab import Datalab
    from speechbrain.pretrained import EncoderClassifier

    SEED = 456  # ensure reproducibility
    feature_extractor = EncoderClassifier.from_hparams(
      "speechbrain/spkrec-xvect-voxceleb",
      # run_opts={"device":"cuda"}  # Uncomment this to run on GPU if you have one (optional)
    )
    import torchaudio

    def extract_audio_embeddings(model, wav_audio_file_path: str) -> tuple:
        """Feature extractor that embeds audio into a vector."""
        signal, fs = torchaudio.load(wav_audio_file_path)  # Reformat audio signal into a tensor
        embeddings = model.encode_batch(
            signal
        )  # Pass tensor through pretrained neural net and extract representation
        return embeddings

    # Extract audio embeddings
    embeddings_list = []
    for i, file_name in enumerate(df[audio_column].tolist()):  # for each .wav file name
        embeddings = extract_audio_embeddings(feature_extractor, file_name)
        embeddings_list.append(embeddings.cpu().numpy())

    embeddings_array = np.squeeze(np.array(embeddings_list))
    print(embeddings_array)
    print("Shape of array: ", embeddings_array.shape)

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    model = LogisticRegression(C=0.01, max_iter=1000, tol=1e-1, random_state=SEED)

    num_crossval_folds = cv_n_folds  # can decrease this value to reduce runtime, or increase it to get better results
    pred_probs = cross_val_predict(
        estimator=model, X=embeddings_array, y=df[label_column].values, cv=num_crossval_folds, method="predict_proba"
    )

    from sklearn.metrics import accuracy_score

    predicted_labels = pred_probs.argmax(axis=1)
    cv_accuracy = accuracy_score(df[label_column].values, predicted_labels)
    print(f"Cross-validated estimate of accuracy on held-out data: {cv_accuracy}")

    lab = Datalab(df, label_name=label_column)
    lab.find_issues(pred_probs=pred_probs, issue_types={label_column:{}})
    lab.report()
    label_issues = lab.get_issues(label_column)
    rv = {}
    rv["label issues"] = label_issues
    return rv

if __name__=="__main__":
    import pandas as pd
    path = r"wav_path.csv"
    df = pd.read_csv(path)
    audio_column = "wav_audio_file_path"
    label_column = "label"
    rv = outlier_detection_audio_pipeline(df,audio_column,label_column)
    
