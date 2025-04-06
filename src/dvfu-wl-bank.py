import copy
import random
import time
import warnings
from typing import Tuple

import kagglehub
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from plotting import plot_unlearning_findings, plot_report_diff, resource_usage

TARGET_SM = "shared_ab"

warnings.simplefilter(action='ignore', category=FutureWarning)

########################################### Constants ###########################################
BANK_MARKETING = "Bank-Marketing"
BATCH_SIZE = 500
PARTIES = 4
IID_CASE = "IID-Case"

# BANK
feature_spec_non_iid_bank = {
    0: ({"job": [-0. - 0.18888962268829346, ]}, 0.3),
    1: ({"job": [-0. - 0.18888962268829346, ]}, 0.7),
    2: ({"job": [-0. - 0.18888962268829346, ]}, 0.2),
    3: ({"job": [-0. - 0.18888962268829346, ]}, 0.3),
    4: ({"job": [-0. - 0.18888962268829346, ]}, 0.1),
}

features_spec_extreme_non_iid_bank = {
    0: ({"job": [-0. - 0.18888962268829346, ]}, 0.0),
    1: ({"job": [-0. - 0.18888962268829346, ]}, 1.0),
    2: ({"job": [-0. - 0.18888962268829346, ]}, 0.0),
    3: ({"job": [-0. - 0.18888962268829346, ]}, 0.0),
    4: ({"job": [-0. - 0.18888962268829346, ]}, 0.0),
}


# Seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


########################################### Data Loading ###########################################
def load_bank_marketing_data():
    path = kagglehub.dataset_download("yufengsui/portuguese-bank-marketing-data-set")

    df = pd.read_csv(
        "/Users/solodezaldivar/.cache/kagglehub/datasets/yufengsui/portuguese-bank-marketing-data-set/versions/1/bank_cleaned.csv",
        sep=',')
    print(df.head())

    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop(columns=['response'])

    df = df[['age', 'default', 'balance', 'job', 'marital', 'education', 'housing', 'loan', 'day', 'month', 'duration',
             'campaign', 'pdays', 'previous', 'poutcome', 'response_binary']]
    print(df.head())
    counts_job = df['job'].value_counts()
    counts_marital = df['marital'].value_counts()
    for value, count in counts_job.items():
        print(f"{value}: {count}")
    for value, count in counts_marital.items():
        print(f"{value}: {count}")

    target_column = "response_binary"

    label_encoders = {}

    for col in df.select_dtypes(include=["object"]).columns:
        if col != target_column:
            unique_vals = df[col].unique()
            # "yes"/"no", map directly
            if len(unique_vals) == 2 and set(unique_vals) == {'yes', 'no'}:
                df[col] = df[col].map({'no': 0, 'yes': 1})
            else:
                # For non-binary categorical features, use LabelEncoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
    # find out which one is job

    # for the nonIID and extreme scenarios both job and education will be used
    # we need to find median of both values in order to better be able to determine where to draw the line of the buckets

    # Split features and target

    X = df.drop(target_column, axis=1).values
    y = df[target_column].values

    # Standardize the features
    scaler = StandardScaler()
    # for row in X:
    #     print(row)
    X = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # Wrap tensors in TensorDataset objects
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset


########################################### Model Definitions ###########################################
class SharedModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=10, num_classes=2):
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size * 2, num_classes)  # why times 2 here TODO

    def forward(self, x):
        return self.relu(self.fc1(x))

    def predict(self, concatenated_hidden):
        return self.fc_out(concatenated_hidden)


class GlobalModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=4, num_classes=2):
        super(GlobalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size * 5, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        return self.relu(self.fc1(x))

    def predict(self, hidden_outs):
        return self.fc_out(hidden_outs)


############################################ Tracking Resources ############################################
def track_resource_usage(epoch, phase="Training"):
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[{phase} - Epoch {epoch + 1}] CPU Memory: {cpu_mem:.2f} GB, GPU Memory: {gpu_mem:.2f} GB")
    return cpu_mem


################################# Unlearning Metrics #################################
def membership_inference_attack(global_models, dataloader, unlearning_step=False):
    for gm in global_models:
        gm.eval()
    with torch.no_grad():
        data, labels = next(iter(dataloader))
        if unlearning_step:
            # Create a mask that zeros out indices where index % 4 == 3 (i.e. partition 4)
            # mask = torch.tensor([ (j % 4 != 1) for j in range(15) ],device=data.device, dtype=torch.bool)
            # data = data * mask
            mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=data.device, dtype=torch.bool)
            data = data * mask
        # Compute hidden outputs and combine them.
        hidden_outputs = generate_hidden_outputs(global_models, data, data.device)
        combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

        # Compute predictions by aggregating outputs from each model.
        outputs = torch.zeros(data.size(0), 2, device=data.device)
        for model in global_models:
            outputs += model.predict(combined_hidden_outputs)
        outputs /= len(global_models)
        loss = F.cross_entropy(outputs, labels)
        print(f"Loss after forgetting parties features: {loss.item():.4f}")
        return loss.item()


def confidence_score_difference(global_models, train_loader, test_loader, unlearning_step=False):
    for gm in global_models:
        gm.eval()
    with torch.no_grad():
        train_data, _ = next(iter(train_loader))
        test_data, _ = next(iter(test_loader))
        device = train_data.device
        if unlearning_step:
            # Create a mask that zeros out indices where index % 4 == 3 (i.e. partition 4)
            mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=device, dtype=torch.bool)
            train_data = train_data * mask
            test_data = test_data * mask
            # train_data = torch.cat((train_data
            # , train_data[:, 6:]), dim=1)
            # test_data = torch.cat((test_data[:, :3], test_data[:, 6:]), dim=1)

        hidden_client_outs = generate_hidden_outputs(global_models, train_data.float(), device)
        combined_hidden_client_outputs = torch.cat(hidden_client_outs, dim=1)
        hidden_unseen_outs = generate_hidden_outputs(global_models, test_data.float(), device)
        combined_hidden_unseen_outs = torch.cat(hidden_unseen_outs, dim=1)
        outputs_train = torch.zeros(train_data.size(0), 2, device=device)
        outputs_unseen = torch.zeros(test_data.size(0), 2, device=device)
        for gm in global_models:
            outputs_train += gm.predict(combined_hidden_client_outputs)
            outputs_unseen += gm.predict(combined_hidden_unseen_outs)
        outputs_train /= len(global_models)
        outputs_unseen /= len(global_models)

        client_confidence = F.softmax(outputs_train, dim=1).max(dim=1)[0].mean().item()

        unseen_confidence = F.softmax(outputs_unseen, dim=1).max(dim=1)[0].mean().item()

        print(f"Confidence on forgotten client data: {client_confidence:.4f}, Unseen data: {unseen_confidence:.4f}")

        return client_confidence, unseen_confidence


def adversarial_mia_attack(global_models, train_loader, test_loader, device, num_epochs=5, unlearning_step=False):
    for gm in global_models:
        gm.eval()

    labels = torch.cat((torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE))).to(device)
    with torch.no_grad():
        training_data, _ = next(iter(train_loader))
        unseen_data, _ = next(iter(test_loader))

        if unlearning_step:
            training_data = training_data.clone()
            unseen_data = unseen_data.clone()
            mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=device, dtype=torch.bool)

            training_data = training_data * mask
            unseen_data = unseen_data * mask

        hidden_client_outs = generate_hidden_outputs(global_models, training_data.float(), device)
        combined_hidden_client_outputs = torch.cat(hidden_client_outs, dim=1)
        hidden_unseen_outs = generate_hidden_outputs(global_models, unseen_data.float(), device)
        combined_hidden_unseen_outs = torch.cat(hidden_unseen_outs, dim=1)
        outputs_train = torch.zeros(training_data.size(0), 2, device=device)
        outputs_unseen = torch.zeros(unseen_data.size(0), 2, device=device)
        for gm in global_models:
            outputs_train += gm.predict(combined_hidden_client_outputs)
            outputs_unseen += gm.predict(combined_hidden_unseen_outs)
        outputs_train /= len(global_models)
        outputs_unseen /= len(global_models)

    attack_data = torch.cat((outputs_train, outputs_unseen), dim=0)
    dataset = TensorDataset(attack_data, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    attack_model = nn.Sequential(nn.Linear(attack_data.size(1), 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()).to(
        device)
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    attack_model.train()
    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = attack_model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    attack_model.eval()
    with torch.no_grad():
        attack_predictions = (attack_model(attack_data.to(device)) > 0.5).float().cpu()
    attack_acc = (attack_predictions == labels.cpu()).float().mean().item()
    print(f"Membership inference attack accuracy: {attack_acc:.4f}")
    return attack_acc


################################## DevertiFL Shared Model Training (Code from DevertiFL repo) ##################################
def generate_hidden_outputs(models, data, device):
    hidden_outputs = []
    for model in models:
        hidden_output = model(data.to(device))
        hidden_outputs.append(hidden_output)
    return hidden_outputs


def train_shared_models(models, device, train_loaders, optimizers, epoch_total, input_size):
    total_resources_shared_model_training = 0
    final_epoch_accuracy = None  # Will hold the accuracy of the final epoch

    # Set all models to training mode
    for model in models:
        model.train()

    # Iterate over each DataLoader (each representing a party's data loader)
    for loader in train_loaders:
        for epoch in range(epoch_total):
            start_time = time.time()
            correct = 0
            total = 0

            # Iterate over each batch in the loader
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)  # Flatten the image

                # Generate hidden outputs from each shared model
                hidden_outputs = generate_hidden_outputs(models, data, device)
                combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

                # Initialize variables to accumulate loss and outputs
                losses = []
                outputs_sum = None

                # For each model, compute predictions and loss
                for i, model in enumerate(models):
                    output = model.predict(combined_hidden_outputs)
                    loss = nn.CrossEntropyLoss()(output, target)
                    losses.append(loss)

                    # Aggregate outputs: here we sum the logits from each model
                    if outputs_sum is None:
                        outputs_sum = output
                    else:
                        outputs_sum = outputs_sum + output

                total_loss = sum(losses)
                total_loss.backward()

                # Update each model's parameters with its corresponding optimizer
                for i, model in enumerate(models):
                    optimizer = optimizers[i]
                    optimizer.step()
                    optimizer.zero_grad()

                # Compute predictions from the aggregated outputs
                preds = torch.argmax(outputs_sum, dim=1)
                correct += (preds == target).sum().item()
                total += target.size(0)

            # Compute accuracy for the epoch
            epoch_accuracy = (correct / total * 100.0) if total > 0 else 0.0
            final_epoch_accuracy = epoch_accuracy

            epoch_time = time.time() - start_time
            total_resources_shared_model_training += track_resource_usage(epoch_time, "Shared Models Training")
            print(
                f"Shared Model [{epoch + 1}/{epoch_total}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return models, total_resources_shared_model_training, final_epoch_accuracy

# Selective gradient exchange function - (Code from DevertiFL repo)
def selective_exchange_gradients(models, hidden_size):
    num_models = len(models)
    total_hidden_size = hidden_size * num_models
    param_indices = [0]
    cumulative_index = 0
    for i in range(num_models):
        cumulative_index += total_hidden_size
        param_indices.append(cumulative_index)

    for seg in range(num_models):
        start = param_indices[seg]
        end = param_indices[seg + 1]
        for param_idx in range(start, end):
            grads = []
            for model in models:
                model_params = list(model.parameters())
                if param_idx < len(model_params) and model_params[param_idx].grad is not None:
                    grads.append(model_params[param_idx].grad)
            if grads:
                avg_grad = torch.stack(grads).mean(dim=0)
                for model in models:
                    model_params = list(model.parameters())
                    if param_idx < len(model_params):
                        model_params[param_idx].grad = avg_grad.clone()
    final_start = param_indices[-1]
    if final_start < len(list(models[0].parameters())):
        for param_idx in range(final_start, len(list(models[0].parameters()))):
            grads = []
            for model in models:
                model_params = list(model.parameters())
                if param_idx < len(model_params) and model_params[param_idx].grad is not None:
                    grads.append(model_params[param_idx].grad)
            if grads:
                avg_grad = torch.stack(grads).mean(dim=0)
                for model in models:
                    if param_idx < len(list(model.parameters())):
                        list(model.parameters())[param_idx].grad = avg_grad.clone()


def vertical_partition_with_feature_from_data(dataset, num_participants, fixed_size=784, feature_spec=None,
                                              attr_indices=None):
    partitions = [[] for _ in range(num_participants)]

    for data, label in dataset:
        data = data.view(-1)
        if data.shape[0] > fixed_size:
            data = data[:fixed_size]
        elif data.shape[0] < fixed_size:
            data = torch.cat([data, torch.zeros(fixed_size - data.shape[0], dtype=data.dtype, device=data.device)])

        # Create a zeroed copy for each partition.
        partition_images = [torch.zeros_like(data) for _ in range(num_participants)]
        for i in range(fixed_size):
            participant = i % num_participants
            partition_images[participant][i] = data[i]

        # Apply feature_spec filtering based on attributes extracted from data.
        for p in range(num_participants):
            meets_conditions = True
            if feature_spec is not None and p in feature_spec and attr_indices is not None:
                conditions, keep_prob = feature_spec[p]
                for key, cond_value in conditions.items():
                    attr_index = attr_indices.get(key)
                    if attr_index is None:
                        meets_conditions = False
                        break
                    # Extract the attribute value from data and convert to a Python scalar.
                    attr_val = data[attr_index].item()
                    # print(f"whaaaaat {attr_val, cond_value}")
                    # If the condition is specified as a list/tuple, check if any value is close.
                    if isinstance(cond_value, (list, tuple)):
                        if not any(
                                np.isclose(attr_val, v, atol=1e-6) if isinstance(v, (int, float)) else attr_val == v for
                                v in cond_value):
                            meets_conditions = False
                            break
                    else:
                        if isinstance(cond_value, (int, float)):
                            if not np.isclose(attr_val, cond_value, atol=1e-6):
                                meets_conditions = False
                                break
                        else:
                            if attr_val != cond_value:
                                meets_conditions = False
                                break
                if not meets_conditions or np.random.rand() > keep_prob:
                    continue  # Skip adding this sample for partition p.
            partitions[p].append((partition_images[p], label))

    # Wrap each partition's list into a DataLoader.
    dataloaders = []
    for p in range(num_participants):
        if partitions[p]:
            data_tensors, labels = zip(*partitions[p])
            dataset_part = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
            dataloader = DataLoader(dataset_part, batch_size=500, shuffle=True)
            dataloaders.append(dataloader)
        else:
            print(f"Warning: Partition {p} is empty and will be skipped.")
    return dataloaders


############################################  Knowledge Distillation (WIP) ############################################
def train_distilled(teacher_model, train_loader, device, temperature=2.0, num_epochs=5):
    student_model = GlobalModel(input_size=588, hidden_size=260, num_classes=10)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    criterion = nn.KLDivLoss(reduction='batchmean')

    teacher_model.eval()
    student_model.train()

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs
            inputs = torch.cat((inputs[:, :196], inputs[:, 392:]), dim=1)  # Keep only m1, m3, and m4

            with torch.no_grad():
                teacher_logits = teacher_model(inputs) / temperature
            student_logits = student_model(inputs)
            loss = criterion(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print("Student model trained using knowledge distillation (removing memroy of m2).")
    return student_model


############################### Aggregate fc_out Weights from Shared Models ###############################
def aggregate_fc_out_weights(state_dicts, model_scores=None) -> Tuple[Tensor, Tensor]:
    weights = [sd['fc_out.weight'] for sd in state_dicts]
    biases = [sd['fc_out.bias'] for sd in state_dicts]

    if model_scores is None:
        normalized_weights = [1.0 / len(state_dicts) for _ in state_dicts]
    else:
        total_score = sum(model_scores)
        normalized_weights = [score / total_score for score in model_scores]

    aggregated_weights = sum((w * param for w, param in zip(normalized_weights, weights)), torch.zeros_like(weights[0]))

    aggregated_bias = sum((w * param for w, param in zip(normalized_weights, biases)), torch.zeros_like(biases[0]))
    return aggregated_weights, aggregated_bias


###################################### Fine-Tuning Global Model on Raw MNIST Data ######################################
def train_global_model(global_model: GlobalModel, train_loader, device, num_epochs=1):
    global_model_cpu = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    global_model.to(device)
    global_model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs

            outputs = global_model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - start_time
        global_model_cpu += track_resource_usage(epoch, "Global Model Training")
        print(f"Global Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")
    return global_model, global_model_cpu


######################################### Evaluation #########################################
def evaluate_global_model(global_model: GlobalModel, test_loader: DataLoader, device,
                          unlearning_step=False):
    global_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not unlearning_step:
                outputs = global_model(data)
            else:
                mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=data.device,
                                    dtype=torch.bool)
                data = data * mask  # keep only m1, m3, m4 and m5
                outputs = global_model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average="macro")
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    report = classification_report(all_targets, all_preds, digits=4, output_dict=True)
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)
    return accuracy, f1, report


def evaluate_shared_model(global_model: GlobalModel, test_loader: DataLoader, device, shared_models=None,
                          unlearning_step=False):
    global_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if not unlearning_step:
                outputs = global_model(data)
            else:
                mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=data.device,
                                    dtype=torch.bool)
                data = data * mask  # keep only m1, m3, m4 and m5
                outputs = global_model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    f1 = f1_score(all_targets, all_preds, average="macro")
    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_targets))
    report = classification_report(all_targets, all_preds, digits=4, output_dict=True)
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(report)
    return accuracy, f1, report


###################################### Fine-Tuning Global Model on Remaining Data #####################################


############################### Forget step direct ###############################
def average_state_dicts(state_dicts):
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        averaged_state_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return averaged_state_dict


def forget_shared_model_direct(forget_name, shared_models, device, model_scores, rfs):
    all_names = ["shared_ab", "shared_ac", "shared_ad", "shared_ae"]

    remaining_models = []
    remaining_models_scores = []
    for name, model, score in zip(all_names, shared_models, model_scores):
        if name != forget_name:
            remaining_models.append(model)
            remaining_models_scores.append(score)

    print(f"Remaining models after forgetting {forget_name}: {len(remaining_models)}: {remaining_models}")
    if len(remaining_models) == 0:
        print("No remaining models available. Cannot unlearn.")
        return None

    # aggregate fc_out weights and biases from the remaining models
    # remaining_state_dicts = [sm[0].state_dict() for sm in remaining_models]
    remaining_state_dicts = [average_state_dicts([sm[0].state_dict(), sm[1].state_dict()]) for sm in remaining_models]

    new_agg_weight, new_agg_bias = aggregate_fc_out_weights(remaining_state_dicts, remaining_models_scores)
    new_agg_weight_fc1, new_agg_bias_fc1 = aggregate_fc1_weights(remaining_state_dicts)

    # new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=15, hidden_size=4, num_classes=2).to(device)
    if not rfs:
        print("Adding pretrained weights")
        new_global_model.fc1.weight.data.copy_(new_agg_weight_fc1)
        new_global_model.fc1.bias.data.copy_(new_agg_bias_fc1)

        new_global_model.fc_out.weight.data.copy_(new_agg_weight)
        new_global_model.fc_out.bias.data.copy_(new_agg_bias)

    print("Global model fc_out updated using direct average of remaining shared models after unlearning", forget_name)
    return new_global_model


def compare_state_dicts(model1, model2, atol=1e-6, rtol=1e-5):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    # Check if both models have the same keys.
    if keys1 != keys2:
        print("The models have different state_dict keys!")
        print("Keys only in model1:", keys1 - keys2)
        print("Keys only in model2:", keys2 - keys1)
    else:
        print("Both models have the same state_dict keys.")

    differences = {}

    # Iterate over each key and compare the tensors.
    for key in keys1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]

        if tensor1.shape != tensor2.shape:
            differences[key] = f"Different shapes: {tensor1.shape} vs {tensor2.shape}"
        else:
            if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
                max_diff = (tensor1 - tensor2).abs().max().item()
                differences[key] = f"Max absolute difference: {max_diff:.6f}"

    if differences:
        print("Differences found in the following parameters:")
        for key, diff in differences.items():
            print(f"{key}: {diff}")
    else:
        print("All parameters are equal within the given tolerance.")

    return differences


############################################ Main Process ############################################
def dvfu_wf_framework(scenario: str, dataset_name: str, retraining_from_scratch: bool):
    print(
        f"######################### Scenario: {scenario} for {dataset_name} with retraining from scratch: {retraining_from_scratch} ######################### ")
    start_time = time.time()
    process = psutil.Process()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = load_bank_marketing_data()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # partition the dataset into 4 disjoint subsets
    fixed_size_shared_model = 6
    dataloaders_shared_models = partition_dataset(train_dataset, scenario, 5, feature_spec_non_iid_bank,
                                                  features_spec_extreme_non_iid_bank, fixed_size_shared_model)
    fixed_size_global_model = 15
    dataloaders_global_model = partition_dataset(train_dataset, scenario, 5, feature_spec_non_iid_bank,
                                                 features_spec_extreme_non_iid_bank, fixed_size_global_model)

    print("Partitioning complete")

    shared_ab, shared_ac, shared_ad, shared_ae, total_cpu_usage_sm, model_scores = get_shared_models(
        dataloaders_shared_models, device)

    print("SharedModels training complete.")
    avg_shared_ab = average_state_dicts([shared_ab[0].state_dict(), shared_ab[1].state_dict()])
    avg_shared_ac = average_state_dicts([shared_ac[0].state_dict(), shared_ac[1].state_dict()])
    avg_shared_ad = average_state_dicts([shared_ad[0].state_dict(), shared_ad[1].state_dict()])
    avg_shared_ae = average_state_dicts([shared_ae[0].state_dict(), shared_ae[1].state_dict()])

    avg_dicts = [avg_shared_ab, avg_shared_ac, avg_shared_ad, avg_shared_ae]

    state_dicts = []
    for i in range(len(avg_dicts)):
        if model_scores[i] > 0.0:
            state_dicts.append(avg_dicts[i])

    print(f"State_dicts {len(state_dicts)}")
    fc_out_weights, fc_out_biases = aggregate_fc_out_weights(state_dicts, model_scores)
    fc1_weights, fc1_biases = aggregate_fc1_weights(state_dicts)

    global_model = create_global_model_agg_weights(fc_out_biases, fc_out_weights, fc1_weights, fc1_biases, False)
    global_models = [copy.deepcopy(global_model) for i in range(5)]
    optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(global_models)}

    print("Fine-tuning Global Model on raw %s training data" % dataset_name)

    global_models = fine_tune_gm(global_models, device, dataloaders_global_model, optimizers, 5, 5, False, 5)

    f1_before, acc_before, report_before = evaluate_gm(global_models, device, test_loader, False)
    print(f"Accuracy: {acc_before}, F1-Score: {f1_before}")
    print(report_before)

    loss_before = membership_inference_attack(global_models, train_loader)
    confi_forgotten_before, confi_unseen_before = confidence_score_difference(global_models, train_loader, test_loader)
    mia_acc_before = adversarial_mia_attack(global_models, train_loader, test_loader, device)

    remaining_shared_models = [shared_ab, shared_ac, shared_ad]

    new_global_model = forget_shared_model_direct("%s" % TARGET_SM, remaining_shared_models, device, model_scores,
                                                  retraining_from_scratch)

    new_global_models = [copy.deepcopy(new_global_model) for i in range(5)]
    new_optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(new_global_models)}

    new_global_models = fine_tune_on_remaining_data(new_global_models, device, dataloaders_global_model, new_optimizers,
                                                    5, 2, 1)

    print("\nEvaluating Global Model after unlearning '%s'..." % TARGET_SM)

    f1_after, acc_after, report_after = evaluate_gm(new_global_models, device, test_loader, True)

    print("Running Unlearning Verification after Forgetting...")
    loss_after = membership_inference_attack(new_global_models, train_loader, unlearning_step=True)
    confi_forgotten_after, confi_unseen_after = confidence_score_difference(new_global_models, train_loader,
                                                                            test_loader, unlearning_step=True)
    mia_acc_after = adversarial_mia_attack(new_global_models, train_loader, test_loader, device, unlearning_step=True)

    total_time = time.time() - start_time
    cpu_mem = process.memory_info().rss / (1024.0 ** 3)
    print(f"Overall CPU Memory Usage: {cpu_mem:.2f} GB, Total Time: {total_time:.2f} seconds")

    # if scenario == EXTREME_NON_IID_CASE:
    #     plot_report_diff(report_before, report_after, dataset_name, EXTREME_NON_IID_CASE, retraining_from_scratch)
    #     plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
    #                              confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
    #                              dataset_name,
    #                              EXTREME_NON_IID_CASE, retraining_from_scratch)
    # elif scenario == NON_IID_CASE:
    #     plot_report_diff(report_before, report_after, dataset_name, NON_IID_CASE, retraining_from_scratch)
    #     plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
    #                              confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
    #                              dataset_name,
    #                              NON_IID_CASE, retraining_from_scratch)

    plot_report_diff(report_before, report_after, dataset_name, IID_CASE, retraining_from_scratch)
    plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                             confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                             dataset_name,
                             IID_CASE, retraining_from_scratch)

    cpu_mem_wf, total_time_wf = cpu_mem, total_time
    print("************* Now Retrain from Scratch *************")
    global_model = GlobalModel(input_size=15, hidden_size=4, num_classes=2)

    global_models = [copy.deepcopy(global_model) for i in range(5)]
    optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(global_models)}

    global_models = fine_tune_on_remaining_data(global_models, device, dataloaders_global_model, optimizers, 5,
                                                5, 5)
    f1_after, acc_after, report_after = evaluate_gm(global_models, device, test_loader, True)
    print(f"Accuracy: {acc_before}, F1-Score: {f1_before}")
    print(report_before)

    loss_after = membership_inference_attack(global_models, train_loader)
    confi_forgotten_after, confi_unseen_after = confidence_score_difference(global_models, train_loader, test_loader)
    mia_acc_after = adversarial_mia_attack(global_models, train_loader, test_loader, device)
    total_time = time.time() - start_time
    cpu_mem = process.memory_info().rss / (1024.0 ** 3)
    cpu_mem_rtfs, total_time_rtfs = cpu_mem, total_time
    retrain_from_scratch = True
    print(f"Overall CPU Memory Usage: {cpu_mem:.2f} MB, Total Time: {total_time:.2f} seconds")

    # if scenario == EXTREME_NON_IID_CASE:
    #     plot_report_diff(report_before, report_after, dataset_name, EXTREME_NON_IID_CASE, retraining_from_scratch)
    #     plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
    #                              confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
    #                              dataset_name,
    #                              EXTREME_NON_IID_CASE, retraining_from_scratch)
    # elif scenario == NON_IID_CASE:
    #     plot_report_diff(report_before, report_after, dataset_name, NON_IID_CASE, retraining_from_scratch)
    #     plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
    #                              confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
    #                              dataset_name,
    #                              NON_IID_CASE, retraining_from_scratch)

    plot_report_diff(report_before, report_after, dataset_name, IID_CASE, retraining_from_scratch)
    plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                             confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                             dataset_name,
                             IID_CASE, retraining_from_scratch)

    return cpu_mem_wf, total_time_wf, cpu_mem_rtfs, total_time_rtfs


def aggregate_fc1_weights(state_dicts):
    weights_list = [sd['fc1.weight'] for sd in state_dicts]
    biases_list = [sd['fc1.bias'] for sd in state_dicts]  # each is (10,)

    W_shared = torch.stack(weights_list, dim=0).mean(dim=0)  # shape: (10,6)
    b_shared = torch.stack(biases_list, dim=0).mean(dim=0)  # shape: (10,)

    W_img = W_shared.unsqueeze(0).unsqueeze(0)
    W_interpolated = F.interpolate(W_img, size=(4, 15), mode='bilinear', align_corners=False)
    W_global = W_interpolated.squeeze(0).squeeze(0)

    b_img = b_shared.unsqueeze(0).unsqueeze(0)
    b_interpolated = F.interpolate(b_img, size=4, mode='linear', align_corners=False)
    b_global = b_interpolated.squeeze()

    return W_global, b_global


def create_global_model_agg_weights(agg_bias, agg_weight, fc1_weight, fc1_bias, rfs):
    global_model = GlobalModel(input_size=15, hidden_size=4, num_classes=2)
    if rfs:
        print("Retraining from scratch, non eed to aggregate classifier weights...")
        return global_model  # do not share shared model weights

    global_model.fc1.weight.data.copy_(fc1_weight)
    global_model.fc1.bias.data.copy_(fc1_bias)

    global_model.fc_out.weight.data.copy_(agg_weight)
    global_model.fc_out.bias.data.copy_(agg_bias)
    print("Global model fc_out initialized with aggregated shared model fc_out weights.")
    return global_model


def get_shared_models(dataloaders, device, federated_rounds=5):
    shared_ab = [SharedModel(), SharedModel()]  # m1 and m2
    shared_ac = [SharedModel(), SharedModel()]  # m1 and m3
    shared_ad = [SharedModel(), SharedModel()]  # m1 and m4
    shared_ae = [SharedModel(), SharedModel()]  # m1 and m5
    total_resources_shared_model_training = 0
    model_scores = []
    print("Training SharedModels Deverti-FL Style")

    for federated_round in range(federated_rounds):
        start_time = time.time()
        shared_ab, total_resource_ab, ab_acc = train_shared_models(shared_ab, device, [dataloaders[0], dataloaders[1]],
                                                                   [optim.Adam(sm.parameters(), lr=0.001) for sm in
                                                                    shared_ab], 5,
                                                                   6)
        shared_ac, total_resource_ac, ac_acc = train_shared_models(shared_ac, device, [dataloaders[0], dataloaders[2]],
                                                                   [optim.Adam(sm.parameters(), lr=0.001) for sm in
                                                                    shared_ac], 5,
                                                                   6)
        shared_ad, total_resource_ad, ad_acc = train_shared_models(shared_ad, device, [dataloaders[0], dataloaders[3]],
                                                                   [optim.Adam(sm.parameters(), lr=0.001) for sm in
                                                                    shared_ad], 5,
                                                                   6)
        shared_ae, total_resource_ae, ae_acc = train_shared_models(shared_ae, device, [dataloaders[0], dataloaders[3]],
                                                                   [optim.Adam(sm.parameters(), lr=0.001) for sm in
                                                                    shared_ad], 5,
                                                                   6)
        total_resources_shared_model_training += total_resource_ab + total_resource_ac + total_resource_ad
        selective_exchange_gradients(shared_ab, 6)
        selective_exchange_gradients(shared_ac, 6)
        selective_exchange_gradients(shared_ad, 6)
        selective_exchange_gradients(shared_ae, 6)
        federated_round_time = time.time() - start_time
        print(
            f"Federated Rounds [{federated_round + 1}/{federated_rounds}] - Time: {federated_round_time:.2f}s, Resource Consumption: {total_resources_shared_model_training:.4f} GB")
        print(f"Accuracy: {ab_acc}, {ac_acc}, {ad_acc}, {ae_acc}")
        model_scores = [ab_acc, ac_acc, ad_acc, ae_acc]
    return shared_ab, shared_ac, shared_ad, shared_ae, total_resources_shared_model_training, model_scores


def partition_dataset(dataset, scenario, num_partitions=4, label_spec_non_iid=None, label_spec_extreme_non_iid=None,
                      fixed_size=6):
    # if scenario == EXTREME_NON_IID_CASE:
    #     dataloaders = vertical_partition_with_feature_from_data(dataset, num_partitions, fixed_size, label_spec_extreme_non_iid, attr_indices)
    # elif scenario == NON_IID_CASE:
    #     dataloaders = vertical_partition_with_feature_from_data(dataset, num_partitions, fixed_size, label_spec_non_iid, attr_indices)
    dataloaders = vertical_partition_with_feature_from_data(dataset, num_partitions, fixed_size, None, None)
    return dataloaders


def fine_tune_on_remaining_data(models, device, train_loaders, optimizers, epoch, federated_rounds, freeze_rounds=2):
    return fine_tune_gm(models, device, train_loaders, optimizers, epoch, federated_rounds, True, freeze_rounds)


def fine_tune_gm(models, device, train_loaders, optimizers, epoch, federated_rounds, unlearning_step=False,
                 freeze_rounds=2):
    for federated_round in range(1, federated_rounds + 1):
        # Freeze or unfreeze the classifier layer depending on the current round.
        for model in models:
            if federated_round <= freeze_rounds:
                # Freeze the classifier layer.
                for param in model.fc_out.parameters():
                    param.requires_grad = False
            else:
                # Unfreeze the classifier layer.
                for param in model.fc_out.parameters():
                    param.requires_grad = True

        print(f"Federated round {federated_round}/{federated_rounds}: "
              f"{'Classifier Frozen' if federated_round <= freeze_rounds else 'Classifier Unfrozen'}")

        # Accumulate losses over the round (for possible logging or consensus exchange).
        num_models = len(models)
        loss_sum = [0.0 for _ in range(num_models)]
        batch_count = 0

        # Set all models to train mode.
        for model in models:
            model.train()

        # Loop over each partition loader.
        for i, train_loader in enumerate(train_loaders):
            for local_epoch in range(1, epoch + 1):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)

                    # For unlearning, if needed, apply a mask to the data for the forgotten partition.
                    if unlearning_step:
                        # For round-robin, if we want to forget partition 4 (index 3),
                        # we construct a mask that zeros out indices where j % 4 == 3.
                        # mask = torch.tensor([j % 4 != 1 for j in range(15)], device=data.device, dtype=torch.bool)
                        # data = data * mask
                        # data = torch.cat((data[:, :3], data[:, 6:]), dim=1)
                        mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=data.device,
                                            dtype=torch.bool)
                        data = data * mask

                    hidden_outputs = generate_hidden_outputs(models, data, device)
                    combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

                    # Compute individual losses.
                    losses = []
                    for model in models:
                        output = model.predict(combined_hidden_outputs)
                        loss = nn.CrossEntropyLoss()(output, target)
                        losses.append(loss)

                    # Accumulate losses.
                    for j in range(num_models):
                        loss_sum[j] += losses[j].item()
                    batch_count += 1

                    # Average losses and backpropagate.
                    total_loss = sum(losses) / len(losses)
                    total_loss.backward()

                    # Update each model.
                    for j, model in enumerate(models):
                        optimizers[j].step()
                        optimizers[j].zero_grad()

        print(f"Federated round {federated_round} completed with total loss: {total_loss:.6f}")
        selective_exchange_gradients(models, 65)
    return models


def evaluate_gm(models, device, test_loader, unlearning_step=False):
    for model in models:
        model.eval()

    all_preds = []
    all_targets = []
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            if unlearning_step:
                # Create a mask that zeros out indices where index % 4 == 3 (i.e. partition 4)
                mask = torch.tensor([(j == 3 or j == 4 or j == 5) for j in range(15)], device=data.device,
                                    dtype=torch.bool)
                data = data * mask
            hidden_outputs = generate_hidden_outputs(models, data, device)
            combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

            outputs = torch.zeros(data.size(0), 2, device=device)
            for model in models:
                outputs += model.predict(combined_hidden_outputs)
            outputs /= len(models)

            # Calculate loss for reporting (if needed).
            loss = criterion(outputs, target)
            test_loss += loss.item()

            # Compute predictions.
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # Collect predictions and targets for metrics.
            all_preds.extend(pred.cpu().squeeze().tolist())
            all_targets.extend(target.cpu().tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, digits=4, output_dict=True)

    return f1, accuracy, report


if __name__ == "__main__":
    set_seed(5)
    start_time = time.time()
    total_cpu = []
    total_time = []
    cpu_mem_dvfu_wf, time_dvfu_wf, cpu_mem_rtfs, time_rtfs = dvfu_wf_framework(IID_CASE, BANK_MARKETING)
    total_cpu.append((cpu_mem_dvfu_wf, cpu_mem_rtfs))
    total_time.append((time_dvfu_wf, time_rtfs))
    print(f"{BANK_MARKETING} - {IID_CASE}")
    print(f"DVFU-WF: CPU={cpu_mem_dvfu_wf} and time={time_dvfu_wf}")
    print(f"RTFS: CPU={cpu_mem_rtfs} and time={time_rtfs}")
    resource_usage(total_cpu, total_time, BANK_MARKETING)
