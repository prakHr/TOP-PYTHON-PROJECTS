
def outlier_detection_image_pipeline(dataset,label_column,image_column,K = 3,n_epochs=2,patience=2,train_batch_size=64,test_batch_size = 512):
    
    def argCheck_outlier_detection_image_pipeline(dataset,label_column,image_column,K,n_epochs,patience,train_batch_size,test_batch_size):
        if type(K)!=int or K<=0:
            return "either K is not integer or K is <=0!" 
        if type(n_epochs)!=int or n_epochs<=0:
            return "either n_epochs is not integer or n_epochs is <=0!" 
        if type(test_batch_size)!=int or test_batch_size<=0 or test_batch_size>len(dataset):
            return "either test_batch_size is not integer or test_batch_size is <=0 or test_batch_size>len(dataset)!" 
        if type(train_batch_size)!=int or train_batch_size<=0 or train_batch_size>len(dataset):
            return "either train_batch_size is not integer or train_batch_size is <=0 or train_batch_size>len(dataset)!" 
        # cols = list(dataset.columns)
        Columns = [image_column,label_column]
        for Column in Columns:
            try:
                C = dataset[Column]
            except:
                return f"{Column} not present in dataset."
        return True
    
    check = argCheck_outlier_detection_image_pipeline(dataset,label_column,image_column,K,n_epochs,patience,train_batch_size,test_batch_size)
    if not check:
        return check
    
    import warnings
    warnings.filterwarnings("ignore", "Lazy modules are a new feature.*")

    from torch.utils.data import DataLoader, TensorDataset, Subset
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import matplotlib.pyplot as plt

    from tqdm import tqdm
    import math
    import time
    import multiprocessing

    from cleanlab import Datalab
    from datasets import load_dataset

    
    num_classes = len(dataset.features[label_column].names)

    # Convert PIL image to torch tensors
    transformed_dataset = dataset.with_format("torch")


    # Apply transformations
    def normalize(example):
        example[image_column] = (example[image_column] / 255.0).unsqueeze(0)
        return example


    transformed_dataset = transformed_dataset.map(normalize, num_proc=multiprocessing.cpu_count())

    
    torch_dataset = TensorDataset(transformed_dataset[image_column], transformed_dataset[label_column])

    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.ReLU(),
                nn.BatchNorm2d(6),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, 5, bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2, 2),
            )
            self.linear = nn.Sequential(nn.LazyLinear(128), nn.ReLU())
            self.output = nn.Sequential(nn.Linear(128, num_classes))

        def forward(self, x):
            x = self.embeddings(x)
            x = self.output(x)
            return x

        def embeddings(self, x):
            x = self.cnn(x)
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = self.linear(x)
            return x

    # This (optional) cell is hidden from docs.cleanlab.ai

    SEED = 1234  # for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(SEED)


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Method to calculate validation accuracy in each epoch
    def get_test_accuracy(net, testloader):
        net.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)

                # run the model on the test set to predict labels
                outputs = net(images)

                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = 100 * accuracy / total
        return accuracy


    # Method for training the model
    def train(trainloader, testloader, n_epochs, patience):
        model = Net()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters())

        model = model.to(device)

        best_test_accuracy = 0.0

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            start_epoch = time.time()
            running_loss = 0.0

            for _, data in enumerate(trainloader):
                # get the inputs; data is a dict of {"image": images, "label": labels}

                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.detach().cpu().item()

            # Get accuracy on the test set
            accuracy = get_test_accuracy(model, testloader)

            if accuracy > best_test_accuracy:
                best_epoch = epoch

            # Condition for early stopping
            if epoch - best_epoch > patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            end_epoch = time.time()

            print(
                f"epoch: {epoch + 1} loss: {running_loss / len(trainloader):.3f} test acc: {accuracy:.3f} time_taken: {end_epoch - start_epoch:.3f}"
            )
        return model


    # Method for computing out-of-sample embeddings
    def compute_embeddings(model, testloader):
        embeddings_list = []

        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data[0].to(device), data[1].to(device)

                embeddings = model.embeddings(images)
                embeddings_list.append(embeddings.cpu())

        return torch.vstack(embeddings_list)


    # Method for computing out-of-sample predicted probabilities
    def compute_pred_probs(model, testloader):
        pred_probs_list = []

        with torch.no_grad():
            for data in tqdm(testloader):
                images, labels = data[0].to(device), data[1].to(device)

                outputs = model(images)
                pred_probs_list.append(outputs.cpu())

        return torch.vstack(pred_probs_list)

    num_workers = multiprocessing.cpu_count()  # Number of workers for data loaders

    # Create k splits of the dataset
    kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    splits = kfold.split(transformed_dataset, transformed_dataset[label_column])

    train_id_list, test_id_list = [], []

    for fold, (train_ids, test_ids) in enumerate(splits):
        train_id_list.append(train_ids)
        test_id_list.append(test_ids)

    
    pred_probs_list, embeddings_list = [], []
    embeddings_model = None

    for i in range(K):
        print(f"\nTraining on fold: {i+1} ...")

        # Create train and test sets and corresponding dataloaders
        trainset = Subset(torch_dataset, train_id_list[i])
        testset = Subset(torch_dataset, test_id_list[i])

        trainloader = DataLoader(
            trainset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        testloader = DataLoader(
            testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # Train model
        model = train(trainloader, testloader, n_epochs, patience)
        if embeddings_model is None:
            embeddings_model = model

        # Compute out-of-sample embeddings
        print("Computing feature embeddings ...")
        fold_embeddings = compute_embeddings(embeddings_model, testloader)
        embeddings_list.append(fold_embeddings)

        print("Computing predicted probabilities ...")
        # Compute out-of-sample predicted probabilities
        fold_pred_probs = compute_pred_probs(model, testloader)
        pred_probs_list.append(fold_pred_probs)

    print("Finished Training")


    # Combine embeddings and predicted probabilities from each fold
    features = torch.vstack(embeddings_list).numpy()

    logits = torch.vstack(pred_probs_list)
    pred_probs = nn.Softmax(dim=1)(logits).numpy()

    
    indices = np.hstack(test_id_list)
    dataset = dataset.select(indices)

    lab = Datalab(data=dataset, label_name=label_column, image_key=image_column)

    
    lab.find_issues(features=features, pred_probs=pred_probs)
    
    lab.report()
    rv = {}
    label_issues = lab.get_issues(label_column)
    
    label_issues_df = label_issues.query("is_label_issue").sort_values("label_score")
    rv["label_issues_df"] = label_issues_df


    
    outlier_issues_df = lab.get_issues("outlier")
    outlier_issues_df = outlier_issues_df.query("is_outlier_issue").sort_values("outlier_score")
    rv["outlier_issues_df"] = outlier_issues_df

    near_duplicate_issues_df = lab.get_issues("near_duplicate")
    near_duplicate_issues_df = near_duplicate_issues_df.query("is_near_duplicate_issue").sort_values(
        "near_duplicate_score"
    )
    rv["near_duplicate_issues_df"] = near_duplicate_issues_df

    
    dark_issues = lab.get_issues("dark")
    dark_issues_df = dark_issues.query("is_dark_issue").sort_values("dark_score")
    rv["dark_issues_df"] = dark_issues_df

    lowinfo_issues = lab.get_issues("low_information")
    lowinfo_issues_df = lowinfo_issues.query("is_low_information_issue").sort_values(
        "low_information_score"
    )
    rv["lowinfo_issues_df"] = lowinfo_issues_df  
    return rv

if __name__=="__main__":
    from datasets import load_dataset
    dataset = load_dataset("fashion_mnist", split="train")
    label_column = "label"
    image_column = "image"
    rv = outlier_detection_image_pipeline(dataset,label_column,image_column)
    from pprint import pprint
    pprint(rv)

