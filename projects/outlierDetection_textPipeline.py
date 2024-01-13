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

    # from datasets import load_dataset
    # dataset = load_dataset("fashion_mnist",split="train")
    # label_column = "label"
    # image_column = "image"
    # rv = outlier_detection_image_pipeline(dataset,label_column,image_column)
    # from pprint import pprint
    # pprint(rv)




# def outlier_detection_image_pipeline(dataset,label_column,image_column,K = 3,n_epochs = 2,train_batch_size = 64,test_batch_size = 512):
#     from torch.utils.data import DataLoader, TensorDataset, Subset
#     import torch
#     import torch.nn as nn
#     import torch.optim as optim

#     from sklearn.model_selection import StratifiedKFold
#     import numpy as np
#     import matplotlib.pyplot as plt

#     from tqdm import tqdm
#     import math
#     import time
#     import multiprocessing

#     from cleanlab import Datalab
#     from datasets import load_dataset
#     # Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     num_classes = len(dataset.features[label_column].names)
#     # Note: This pulldown content is for docs.cleanlab.ai, if running on local Jupyter or Colab, please ignore it.

    
#     class Net(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.cnn = nn.Sequential(
#                 nn.Conv2d(1, 6, 5),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(6),
#                 nn.MaxPool2d(2, 2),
#                 nn.Conv2d(6, 16, 5, bias=False),
#                 nn.ReLU(),
#                 nn.BatchNorm2d(16),
#                 nn.MaxPool2d(2, 2),
#             )
#             self.linear = nn.Sequential(nn.LazyLinear(128), nn.ReLU())
#             self.output = nn.Sequential(nn.Linear(128, num_classes))

#         def forward(self, x):
#             x = self.embeddings(x)
#             x = self.output(x)
#             return x

#         def embeddings(self, x):
#             x = self.cnn(x)
#             x = torch.flatten(x, 1)  # flatten all dimensions except batch
#             x = self.linear(x)
#             return x
#     # Method to calculate validation accuracy in each epoch
#     def get_test_accuracy(net, testloader):
#         net.eval()
#         accuracy = 0.0
#         total = 0.0

#         with torch.no_grad():
#             for data in testloader:
#                 images, labels = data[image_column].to(device), data[label_column].to(device)

#                 # run the model on the test set to predict labels
#                 outputs = net(images)

#                 # the label with the highest energy will be our prediction
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 accuracy += (predicted == labels).sum().item()

#         # compute the accuracy over all test images
#         accuracy = 100 * accuracy / total
#         return accuracy


#     # Method for training the model
#     def train(trainloader, testloader, n_epochs, patience):
#         model = Net()

#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.AdamW(model.parameters())

#         model = model.to(device)

#         best_test_accuracy = 0.0

#         for epoch in range(n_epochs):  # loop over the dataset multiple times
#             start_epoch = time.time()
#             running_loss = 0.0

#             for _, data in enumerate(trainloader):
#                 # get the inputs; data is a dict of {"image": images, "label": labels}

#                 inputs, labels = data[image_column].to(device), data[label_column].to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward + backward + optimize
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.detach().cpu().item()

#             # Get accuracy on the test set
#             accuracy = get_test_accuracy(model, testloader)

#             if accuracy > best_test_accuracy:
#                 best_epoch = epoch

#             # Condition for early stopping
#             if epoch - best_epoch > patience:
#                 print(f"Early stopping at epoch {epoch + 1}")
#                 break

#             end_epoch = time.time()

#             print(
#                 f"epoch: {epoch + 1} loss: {running_loss / len(trainloader):.3f} test acc: {accuracy:.3f} time_taken: {end_epoch - start_epoch:.3f}"
#             )
#         return model


#     # Method for computing out-of-sample embeddings
#     def compute_embeddings(model, testloader):
#         embeddings_list = []

#         with torch.no_grad():
#             for data in tqdm(testloader):
#                 images, labels = data[image_column].to(device), data[label_column].to(device)

#                 embeddings = model.embeddings(images)
#                 embeddings_list.append(embeddings.cpu())

#         return torch.vstack(embeddings_list)


#     # Method for computing out-of-sample predicted probabilities
#     def compute_pred_probs(model, testloader):
#         pred_probs_list = []

#         with torch.no_grad():
#             for data in tqdm(testloader):
#                 images, labels = data[image_column].to(device), data[label_column].to(device)

#                 outputs = model(images)
#                 pred_probs_list.append(outputs.cpu())

#         return torch.vstack(pred_probs_list)
#     # Convert PIL image to torch tensors
#     transformed_dataset = dataset.with_format("torch")


#     # Apply transformations
#     def normalize(example):
#         example[image_column] = (example[image_column] / 255.0).unsqueeze(0)
#         return example


#     transformed_dataset = transformed_dataset.map(normalize, num_proc=multiprocessing.cpu_count())
#     torch_dataset = TensorDataset(transformed_dataset[image_column], transformed_dataset[label_column])
#     # Number of cross-validation folds. Set to small value here to ensure quick runtimes, we recommend 5 or 10 in practice for more accurate estimates.
#     # Number of epochs to train model for. Set to a small value here for quick runtime, you should use a larger value in practice.
#     patience = 2  # Parameter for early stopping. If the validation accuracy does not improve for this many epochs, training will stop.
#     # Batch size for training
#     # Batch size for testing
#     num_workers = multiprocessing.cpu_count()  # Number of workers for data loaders

#     # Create k splits of the dataset
#     kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
#     splits = kfold.split(transformed_dataset, transformed_dataset[label_column])

#     train_id_list, test_id_list = [], []

#     for fold, (train_ids, test_ids) in enumerate(splits):
#         train_id_list.append(train_ids)
#         test_id_list.append(test_ids)

#     pred_probs_list, embeddings_list = [], []
#     embeddings_model = None

#     for i in range(K):
#         print(f"\nTraining on fold: {i+1} ...")

#         # Create train and test sets and corresponding dataloaders
#         trainset = Subset(torch_dataset, train_id_list[i])
#         testset = Subset(torch_dataset, test_id_list[i])

#         trainloader = DataLoader(
#             trainset,
#             batch_size=train_batch_size,
#             shuffle=False,
#             num_workers=num_workers,
#             pin_memory=True,
#         )
#         testloader = DataLoader(
#             testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
#         )

#         # Train model
#         model = train(trainloader, testloader, n_epochs, patience)
#         if embeddings_model is None:
#             embeddings_model = model

#         # Compute out-of-sample embeddings
#         print("Computing feature embeddings ...")
#         fold_embeddings = compute_embeddings(embeddings_model, testloader)
#         embeddings_list.append(fold_embeddings)

#         print("Computing predicted probabilities ...")
#         # Compute out-of-sample predicted probabilities
#         fold_pred_probs = compute_pred_probs(model, testloader)
#         pred_probs_list.append(fold_pred_probs)

#     print("Finished Training")


#     # Combine embeddings and predicted probabilities from each fold
#     features = torch.vstack(embeddings_list).numpy()

#     logits = torch.vstack(pred_probs_list)
#     pred_probs = nn.Softmax(dim=1)(logits).numpy()

#     indices = np.hstack(test_id_list)
#     dataset = dataset.select(indices)
#     lab = Datalab(data=dataset, label_name=label_column, image_key=image_column)
#     lab.find_issues(features=features, pred_probs=pred_probs)
#     rv = {}

#     label_issues_df =lab.get_issues(label_column)
#     rv["label issues"] = label_issues_df

#     outlier_issues_df = lab.get_issues("outlier")
#     rv["outlier issues"] = outlier_issues_df
    
#     near_duplicate_issues_df = lab.get_issues("near_duplicate")
#     rv["near duplicate issues"] = near_duplicate_issues_df

#     dark_issues_df = lab.get_issues("dark")
#     rv["dark issues"] = dark_issues_df

#     return rv