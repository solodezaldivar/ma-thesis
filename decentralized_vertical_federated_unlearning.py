# TODO: compare time perspective of retraining the whole federation vs only the global model
# TODO: global model is trained through decentralized sgd with consensus on the weights in ring topology
import collections
import concurrent.futures
import copy
from typing import Tuple

import kagglehub
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch import Tensor
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import psutil
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, classification_report
import torch.nn.functional as F
import warnings
import random

from deverti_fl import vertical_partition_rotating
from plotting import plot_unlearning_findings, plot_report_diff, plot_resource_comparison, resource_usage

warnings.simplefilter(action='ignore', category=FutureWarning)

########################################### Constants ###########################################
MNIST = "MNIST"
FASHION_MNIST = "Fashion-MNIST"
BANK_MARKETING = "Bank Marketing"
BATCH_SIZE = 32
PARTIES = 4
non_IID = False
EXTREME_NON_IID_CASE = "extreme-non-IID-Case"
NON_IID_CASE = "non-IID-Case"
IID_CASE = "IID-Case"

# MNIST
label_spec_extreme_non_iid_mnist = {
    0: (5, 0.0),
    1: (5, 0.0),
    2: (5, 0.0),
    3: (5, 1.0),
}

label_spec_normal_non_iid_mnist = {
    0: (5, 0.1),
    1: (5, 0.2),
    2: (5, 0.0),
    3: (5, 0.7),
}

# FMNIST
label_spec_extreme_non_iid_fashion_mnist = {
    0: (5, 0.0),
    1: (5, 0.0),
    2: (5, 0.0),
    3: (5, 1.0),
}

label_spec_normal_non_iid_fashion_mnist = {
    0: (5, 0.1),
    1: (5, 0.2),
    2: (5, 0.0),
    3: (5, 0.7),
}

# Seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For some operations, setting torch.backends may further improve reproducibility:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################################### Data Loading ###########################################
# Fashion-MNIST data
def load_fashion_mnist_data():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    # Preprocessing for the test set (usually no augmentation, only normalization)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    return train_dataset, test_dataset

########################################### Model Definitions ###########################################
class SharedModel(nn.Module):
    def __init__(self, input_size=392, hidden_size=130, num_classes=10):
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        return self.relu(self.fc1(x))

    def predict(self, concatenated_hidden):
        return self.fc_out(concatenated_hidden)


class GlobalModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=65, num_classes=10):
        super(GlobalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size * 4, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        hidden_out = self.relu(self.fc1(x))
        return self.dropout(hidden_out)


    def predict(self, concatenated_hidden):
        return self.fc_out(concatenated_hidden)


############################################ Tracking Resources ############################################
def track_resource_usage(epoch, phase="Training"):
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[{phase} - Epoch {epoch + 1}] CPU Memory: {cpu_mem:.2f} GB, GPU Memory: {gpu_mem:.2f} GB")
    return cpu_mem


################################# Unlearning Metrics #################################
def loss_eval(global_models, dataloader, unlearning_step=False):
    for gm in global_models:
        gm.eval()
    with torch.no_grad():
        data, labels = next(iter(dataloader))
        if unlearning_step:
            # Create a mask that zeros out indices where index % 4 == 3 (i.e. partition 4)
            mask = torch.tensor([ (j % 4 != 3) for j in range(784) ],device=data.device, dtype=torch.bool)
            data = data * mask
            # Compute hidden outputs and combine them.
        hidden_outputs = generate_hidden_outputs(global_models, data, data.device)
        combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

        # Compute predictions by aggregating outputs from each model.
        outputs = torch.zeros(data.size(0), 10, device=data.device)
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
            mask = torch.tensor([ (j % 4 != 3) for j in range(784) ],device=device, dtype=torch.bool)
            train_data = train_data * mask
            test_data = test_data * mask

        hidden_client_outs = generate_hidden_outputs(global_models, train_data.float(), device)
        combined_hidden_client_outputs = torch.cat(hidden_client_outs, dim=1)
        hidden_unseen_outs = generate_hidden_outputs(global_models, test_data.float(), device)
        combined_hidden_unseen_outs = torch.cat(hidden_unseen_outs, dim=1)
        outputs_train = torch.zeros(train_data.size(0), 10, device=device)
        outputs_unseen = torch.zeros(test_data.size(0), 10, device=device)
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
            mask = torch.tensor([i % 4 != 1 for i in range(784)], dtype=torch.bool, device=training_data.device)
            training_data = training_data.clone()
            unseen_data = unseen_data.clone()
            training_data[:, ~mask] = 0
            unseen_data[:, ~mask] = 0



        hidden_client_outs = generate_hidden_outputs(global_models, training_data.float(), device)
        combined_hidden_client_outputs = torch.cat(hidden_client_outs, dim=1)
        hidden_unseen_outs = generate_hidden_outputs(global_models, unseen_data.float(), device)
        combined_hidden_unseen_outs = torch.cat(hidden_unseen_outs, dim=1)
        outputs_train = torch.zeros(training_data.size(0), 10, device=device)
        outputs_unseen = torch.zeros(unseen_data.size(0), 10, device=device)
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


################################## DevertiFL Shared Model Training ##################################
def generate_hidden_outputs(models, data, device):
    hidden_outputs = []
    for model in models:
        hidden_output = model(data.to(device))
        hidden_outputs.append(hidden_output)
    return hidden_outputs


def train_shared_models(models, device, train_loaders, optimizers, epoch_total, is_shared_model=True):
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
            print(f"Shared Model [{epoch + 1}/{epoch_total}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return models, total_resources_shared_model_training, final_epoch_accuracy

# Selective gradient exchange function - Code from DevertiFL repo
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

def partition_dataset_round_robin_order_preserving(dataset, num_partitions, batch_size=500, shuffle=True, fixed_size=784, label_spec=None):

    partitions = [[] for _ in range(num_partitions)]

    # Pre-compute the active indices for each partition. For partition i, these are indices j such that j % num_partitions == i.
    active_indices = [list(range(i, fixed_size, num_partitions)) for i in range(num_partitions)]

    for data, label in dataset:
        for i in range(num_partitions):
            # Optional label filtering logic
            if label_spec is not None and i in label_spec:
                target_label, keep_prob = label_spec[i]
                if keep_prob == 1.0 and label != target_label:
                    continue
                elif label == target_label and np.random.rand() > keep_prob:
                    continue

            new_data = torch.zeros(fixed_size, dtype=data.dtype)
            # Copy only the features for this partition, preserving their original positions.
            indices = active_indices[i]
            new_data[indices] = data[indices]
            partitions[i].append((new_data, label))

    # Wrap each partition's samples into a DataLoader.
    dataloaders = []
    for part in partitions:
        dataset_part = TensorDataset(torch.stack([x for x, _ in part]), torch.tensor([y for _, y in part]))
        loader = DataLoader(dataset_part, batch_size=batch_size, shuffle=shuffle)
        dataloaders.append(loader)

    return dataloaders



# def partition_dataset_bank(dataset, num_partitions, batch_size=500, shuffle=True, fixed_size=392, label_spec=None):
#
#     sample, _ = dataset[0]
#     feature_dim = sample.numel()
#     partition_size = feature_dim // num_partitions
#
#     dataloaders = []
#     partition_info = {}
#
#     for i in range(num_partitions):
#         start = i * partition_size
#         end = (i + 1) * partition_size if i < num_partitions - 1 else feature_dim
#         partition_samples = []
#         for data, label in dataset:
#             if label_spec is not None and i in label_spec:
#                 target_spec, keep_prob = label_spec[i]
#                 meets_spec = True
#                 # Verify each feature in the target specification.
#                 for key, allowed_values in target_spec.items():
#                     if key not in label or label[key] not in allowed_values:
#                         meets_spec = False
#                         break
#                 # Filtering samples based on the keep probability.
#                 if keep_prob == 1.0:
#                     if not meets_spec:
#                         continue
#                 else:
#                     if meets_spec and np.random.rand() > keep_prob:
#                         continue
#             # Create a new tensor of fixed_size and copy the appropriate slice from data.
#             new_data = torch.zeros(fixed_size, dtype=data.dtype)
#             slice_data = data[start:end]
#             new_data[:min(slice_data.numel(), fixed_size)] = slice_data[:fixed_size]
#             partition_samples.append((new_data, label))
#
#         # Create a DataLoader for the partition.
#         dataloader = DataLoader(partition_samples, batch_size=batch_size, shuffle=shuffle)
#         dataloaders.append(dataloader)
#
#         # Verification: count the number of samples and frequency distribution of 'job' and 'education'.
#         job_counter = collections.Counter()
#         education_counter = collections.Counter()
#         for _, label in partition_samples:
#             # Expecting label to be a dictionary with 'job' and 'education'
#             job_counter[label.get('job', 'unknown')] += 1
#             education_counter[label.get('education', 'unknown')] += 1
#         partition_info[i] = {
#             "num_samples": len(partition_samples),
#             "job_distribution": dict(job_counter),
#             "education_distribution": dict(education_counter)
#         }
#
#     # Print verification statistics for each partition.
#     print("Partition Verification Info:")
#     for i, info in partition_info.items():
#         print(f"\nPartition {i}: {info['num_samples']} samples")
#         print("Job distribution:", info["job_distribution"])
#         print("Education distribution:", info["education_distribution"])
#
#     return dataloaders, partition_info


############################################  Knowledge Distillation (WIP) ############################################
# def train_distilled(teacher_model, train_loader, device, temperature=2.0, num_epochs=5):
#     student_model = GlobalModel(input_size=588, hidden_size=260, num_classes=10)
#     optimizer = optim.Adam(student_model.parameters(), lr=0.001)
#     criterion = nn.KLDivLoss(reduction='batchmean')
#
#     teacher_model.eval()
#     student_model.train()
#
#     for epoch in range(num_epochs):
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs
#             inputs = torch.cat((inputs[:, :196], inputs[:, 392:]), dim=1)  # Keep only m1, m3, and m4
#
#             with torch.no_grad():
#                 teacher_logits = teacher_model(inputs) / temperature
#             student_logits = student_model(inputs)
#             loss = criterion(F.log_softmax(student_logits / temperature, dim=1), F.softmax(teacher_logits, dim=1))
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#     print("Student model trained using knowledge distillation (removing memroy of m2).")
#     return student_model


############################### Aggregate fc_out Weights from Shared Models ###############################
def aggregate_fc_out_weights(state_dicts, model_scores=None) -> Tuple[Tensor, Tensor]:
    weights = [sd['fc_out.weight'] for sd in state_dicts]
    biases = [sd['fc_out.bias'] for sd in state_dicts]

    if model_scores is None:
        normalized_weights = [1.0 / len(state_dicts) for _ in state_dicts]
    else:
        # Convert scores to a tensor and compute softmax weights.
        scores_tensor = torch.tensor(model_scores, dtype=torch.float32)
        normalized_weights = F.softmax(scores_tensor, dim=0).tolist()

    aggregated_weights = sum((w * param for w, param in zip(normalized_weights, weights)), torch.zeros_like(weights[0]))
    aggregated_bias = sum((w * param for w, param in zip(normalized_weights, biases)), torch.zeros_like(biases[0]))
    return aggregated_weights, aggregated_bias

def aggregate_fc1_weights_three(shared_models):
    common_weights = []
    unique_weights = []
    common_biases = []
    for model in shared_models:
        W = model.fc1.weight.data  # shape (130, 392)
        b = model.fc1.bias.data    # shape (130,)
        common_weights.append(W[:, :196])
        unique_weights.append(W[:, 196:392])
        common_biases.append(b)

    # Average the common parts (should be similar across models).
    W_common = torch.stack(common_weights, dim=0).mean(dim=0)  # (130, 196)
    # Average the unique parts.
    W_unique = torch.stack(unique_weights, dim=0).mean(dim=0)  # (130, 196)

    # Construct one half of the global fc1 weight by concatenation.
    # For the unique segments, we replicate the averaged unique weight.
    W_half = torch.cat([W_common, W_unique, W_unique, W_unique], dim=1)  # (130, 196+196*3=784)

    # Duplicate W_half vertically to match the global model's output dimension (260).
    global_fc1_weight = torch.cat([W_half, W_half], dim=0)  # (260, 784)

    # For biases, average the shared biases and duplicate.
    avg_bias = torch.stack(common_biases, dim=0).mean(dim=0)  # (130,)
    global_fc1_bias = torch.cat([avg_bias, avg_bias], dim=0)    # (260,)

    return global_fc1_weight, global_fc1_bias


###################################### Fine-Tuning Global Model on Raw MNIST Data ######################################
def train_global_model(global_model: GlobalModel, train_loader, device, num_epochs=1): # the non federated way
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

def compute_attention_weights(param_list):
    scores = [torch.mean(torch.abs(t)) for t in param_list]
    scores_tensor = torch.stack(scores)
    attn_weights = torch.softmax(scores_tensor, dim=0)
    return attn_weights

######################################### Evaluation #########################################
# def evaluate(models, device, test_loader):
#     for model in models:
#         model.eval()
#
#     with torch.no_grad():
#         test_loss = 0
#         correct = 0
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             data = data.view(data.size(0), -1)
#
#             hidden_outputs = generate_hidden_outputs(models, data, device)
#             combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)
#             outputs = torch.zeros(data.size(0), 10, device=device)
#             for model in models:
#                 output = model.predict(combined_hidden_outputs)
#                 outputs += output
#             outputs /= len(models)
#
#             test_loss += nn.CrossEntropyLoss(reduction='sum')(outputs, target).item()
#             pred = outputs.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#         test_loss /= len(test_loader.dataset)
#         accuracy = 100. * correct / len(test_loader.dataset)
#         return accuracy
###################################### Fine-Tuning Global Model on Remaining Data #####################################
# def fine_tune_on_remaining_data(global_model, train_loader, device, num_epochs=5):
#     print(f"Fine-tuning global model on remaining parties")
#     cpu_mem_fine_tune = 0
#     for param in global_model.fc_out.parameters():
#         param.requires_grad = False
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(global_model.parameters(), lr=0.001)
#     global_model.to(device)
#     global_model.train()
#
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         for inputs, labels in train_loader:
#             if inputs.shape[0] == 0:  # Skip batch if empty after filtering
#                 continue
#
#             inputs, labels = inputs.to(device), labels.to(device)
#             inputs = inputs
#
#             inputs = torch.cat((inputs[:, :196], inputs[:, 392:]), dim=1)  # Keep only m1, m3, and m4
#             optimizer.zero_grad()
#             outputs = global_model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#         epoch_time = time.time() - start_time
#         cpu_mem_fine_tune += track_resource_usage(epoch, "Fine-Tuned Global Model Training")
#         print(f"Fine-tune Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")
#
#     return global_model, cpu_mem_fine_tune


############################### Forget step direct ###############################
def forget_shared_model_direct(forget_name, shared_models, device, model_scores, rfs):
    print(f"\nUnlearning: Removing shared model '{forget_name}'...")
    all_names = ["shared_ab", "shared_ac", "shared_ad"]

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
    remaining_state_dicts = [sm[0].state_dict() for sm in remaining_models]
    # remaining_state_dicts = [average_state_dicts([sm[0].state_dict(), sm[1].state_dict()]) for sm in remaining_models]

    new_agg_weight, new_agg_bias = aggregate_fc_out_weights(remaining_state_dicts, remaining_models_scores)

    # new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=784, hidden_size=65, num_classes=10).to(device)
    if rfs:
        print("Retraining from scratch, non eed to aggregate classifier weights...")
        return new_global_model

    print("Adding pretrained weights")
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


def average_state_dicts(state_dicts):
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key] for sd in state_dicts]
        averaged_state_dict[key] = torch.stack(tensors, dim=0).mean(dim=0)
    return averaged_state_dict

# def consensus_parameter_exchange_weighted(models, avg_losses, eps=1e-8):
#     # Compute weights as inverse of loss, then normalize so that they sum to 1.
#     inv_losses = [1.0 / (loss + eps) for loss in avg_losses]
#     total_inv = sum(inv_losses)
#     weights = [w / total_inv for w in inv_losses]
#
#     # Retrieve state dictionaries.
#     state_dicts = [model.state_dict() for model in models]
#     avg_state = {}
#     # For each parameter key, compute a weighted average.
#     for key in state_dicts[0]:
#         avg_state[key] = sum(weights[i] * state_dicts[i][key] for i in range(len(models)))
#
#     # Update each model with the averaged state.
#     for model in models:
#         model.load_state_dict(avg_state)
# def consensus_parameter_exchange(models):
#     """
#     Averages the state dictionaries of all models and loads the averaged
#     parameters back into each model.
#     """
#     # Collect state dicts from all models.
#     state_dicts = [model.state_dict() for model in models]
#     avg_state = {}
#     # Average each parameter by key.
#     for key in state_dicts[0]:
#         avg_state[key] = sum(sd[key] for sd in state_dicts) / len(models)
#     # Update each model with the averaged state.
#     for model in models:
#         model.load_state_dict(avg_state)

############################################ Main Process ############################################
def devertifl_global_model(models, device, train_loaders, optimizers, epoch, federated_rounds, unlearning_step=False):
    for federated_round in range(1, federated_rounds + 1):
        # Accumulate losses per model over the round.
        num_models = len(models)
        loss_sum = [0.0 for _ in range(num_models)]
        batch_count = 0
        for model in models:
            model.train()

        for i, train_loader in enumerate(train_loaders):
            for local_epoch in range(1, epoch + 1):

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    if unlearning_step:
                        mask = torch.tensor([j % 4 != 3 for j in range(784)], device=data.device, dtype=torch.bool)
                        data = data * mask
                    hidden_outputs = generate_hidden_outputs(models, data, device)
                    combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

                    # Compute individual losses for each model.
                    losses = []
                    for model in models:
                        output = model.predict(combined_hidden_outputs)
                        loss = nn.CrossEntropyLoss()(output, target)
                        losses.append(loss)

                    # Compute total loss as average (for backward pass) and backpropagate.
                    total_loss = sum(losses) / len(losses)
                    total_loss.backward()

                    # Update models.
                    for i, model in enumerate(models):
                        optimizers[i].step()
                        optimizers[i].zero_grad()

        print(f"Total loss: {total_loss:.6f}")
        selective_exchange_gradients(models, 65)
    return models

def devertifl_global_model_with_freeze(models, device, train_loaders, optimizers, epoch, federated_rounds, unlearning_step=False, freeze_rounds=2):

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
                        mask = torch.tensor([j % 4 != 3 for j in range(784)], device=data.device, dtype=torch.bool)
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

def evaluate_plain_vertifl(models, device, test_loader, unlearning_step=False):
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
                mask = torch.tensor([ (j % 4 != 3) for j in range(784) ],device=data.device, dtype=torch.bool)
                data = data * mask
            # Compute hidden outputs and combine them.
            hidden_outputs = generate_hidden_outputs(models, data, device)
            combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

            # Compute predictions by aggregating outputs from each model.
            outputs = torch.zeros(data.size(0), 10, device=device)
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


# def pad_data_part(data, partition_index, total_dim=784, active_size=196, mean=None, std=None):
#     """
#     Pads a data part (assumed shape (batch_size, active_size)) to a total dimension (e.g., 784).
#     The active data is placed in the block corresponding to partition_index:
#
#       - If partition_index == 0: fill indices [0:196] with data, zeros elsewhere.
#       - If partition_index == 1: fill indices [196:392] with data, zeros elsewhere.
#       - If partition_index == 2: fill indices [392:588] with data, zeros elsewhere.
#       - If partition_index == 3: fill indices [588:784] with data, zeros elsewhere.
#
#     Then, if mean and std are provided, the entire padded vector is normalized.
#
#     Args:
#         data (Tensor): shape (batch_size, active_size)
#         partition_index (int): which partition (0, 1, 2, or 3).
#         total_dim (int): Total dimension after padding.
#         active_size (int): Size of the active block.
#         mean (Tensor or float, optional): Mean used for normalization.
#         std (Tensor or float, optional): Std used for normalization.
#
#     Returns:
#         padded_data (Tensor): shape (batch_size, total_dim), normalized if mean and std are provided.
#     """
#     batch_size = data.size(0)
#     padded = torch.zeros(batch_size, total_dim, device=data.device, dtype=data.dtype)
#     start = partition_index * active_size
#     end = start + active_size
#     padded[:, start:end] = data
#     if mean is not None and std is not None:
#         padded = (padded - mean) / std
#     return padded


# def vertical_partition_sequential_custom(dataset, boundaries, fixed_size=784, mean=0.0, std=1.0):
#     """
#     Splits each flattened image (a 1D tensor of length fixed_size, e.g. 784) into partitions
#     defined by custom boundaries.
#
#     For each image, for each partition defined by (start, end) in boundaries:
#       - Create a new vector of length fixed_size.
#       - Fill indices [start, end) with the corresponding slice from the image.
#       - The rest of the vector is zeros.
#       - Optionally normalize the vector using the provided mean and std.
#
#     Args:
#         dataset: An iterable of (image, label) pairs, where each image is a 1D tensor of length fixed_size.
#         boundaries (list of tuples): List of (start, end) pairs defining the active region for each partition.
#             For example: [(0,392), (392,522), (522,652), (652,784)]
#         fixed_size (int): The total length of each output vector (e.g., 784).
#         mean (float): Mean for normalization.
#         std (float): Standard deviation for normalization.
#
#     Returns:
#         A list of DataLoader objects, one per partition.
#     """
#     num_partitions = len(boundaries)
#     partitions = [[] for _ in range(num_partitions)]
#
#     for image, label in dataset:
#         # Assume image is a 1D tensor of length fixed_size (e.g., 784).
#         for i, (start, end) in enumerate(boundaries):
#             new_image = torch.zeros(fixed_size, dtype=image.dtype)
#             new_image[start:end] = image[start:end]
#             partitions[i].append((new_image, label))
#
#     dataloaders = []
#     for part in partitions:
#         data_tensors, labels = zip(*part)
#         ds = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
#         loader = DataLoader(ds, batch_size=500, shuffle=True)
#         dataloaders.append(loader)
#
#     return dataloaders

# def vertical_partition_sequential(dataset, num_partitions, fixed_size, active_size=196, mean=0.0, std=1.0):
#     partitions = [[] for _ in range(num_partitions)]
#     for image, label in dataset:
#         # image is assumed to be already flattened to 784.
#         for p in range(num_partitions):
#             # Extract the active block corresponding to this partition.
#             start = p * active_size
#             end = start + active_size
#             # Here data part is image[start:end] of shape (active_size,)
#             data_part = image[start:end].unsqueeze(0)  # shape (1, active_size)
#             # Pad the data part to fixed_size (e.g., 784) and normalize.
#             padded = pad_data_part(data_part, p, total_dim=fixed_size, active_size=active_size, mean=mean, std=std)
#             # Remove the batch dimension.
#             padded = padded.squeeze(0)
#             partitions[p].append((padded, label))
#
#     dataloaders = []
#     for part in partitions:
#         data_tensors, labels = zip(*part)
#         dataset_part = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
#         loader = DataLoader(dataset_part, batch_size=500, shuffle=True)
#         dataloaders.append(loader)
#
#     return dataloaders



def dvfu_framework(scenario: str, dataset_name: str, retraining_from_scratch: bool):
    print(f"######################### Scenario: {scenario} for {dataset_name} with retraining from scratch: {retraining_from_scratch} ######################### ")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name.upper() == MNIST.upper():
        train_dataset, test_dataset = load_mnist_data()
        label_spec_non_iid, label_spec_extreme_non_iid = label_spec_normal_non_iid_mnist, label_spec_extreme_non_iid_mnist

    elif dataset_name.upper() == FASHION_MNIST.upper():
        train_dataset, test_dataset = load_fashion_mnist_data()
        label_spec_non_iid, label_spec_extreme_non_iid = label_spec_normal_non_iid_fashion_mnist, label_spec_extreme_non_iid_fashion_mnist

    # elif dataset_name.upper() == BANK_MARKETING.upper():
    #         train_dataset, test_dataset = load_bank_marketing_data()
    #         label_spec_non_iid, label_spec_extreme_non_iid = label_spec_normal_non_iid_bank, label_spec_extreme_non_iid_bank

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # partition the dataset into 4 disjoint subsets for both shared models and global model
    shared_model_input_size = 392 # 196 * 2
    global_model_input_size = 784
    dataloaders = get_data_loader(train_dataset, scenario, label_spec_non_iid, label_spec_extreme_non_iid, shared_model_input_size)
    dataloaders_global_model = get_data_loader(train_dataset, scenario, label_spec_non_iid, label_spec_extreme_non_iid, global_model_input_size)

    # Initialize three shared models.
    shared_ab, shared_ac, shared_ad, total_cpu_usage_sm, model_scores = get_shared_models(dataloaders, device)

    print("SharedModels training complete.")
    avg_shared_ab = average_state_dicts([shared_ab[0].state_dict(), shared_ab[1].state_dict()])
    avg_shared_ac = average_state_dicts([shared_ac[0].state_dict(), shared_ac[1].state_dict()])
    avg_shared_ad = average_state_dicts([shared_ad[0].state_dict(), shared_ad[1].state_dict()])


    state_dicts = [avg_shared_ab, avg_shared_ac, avg_shared_ad]
    agg_weight, agg_bias = aggregate_fc_out_weights(state_dicts, model_scores)
    fc1_weight, fc1_bias = aggregate_fc1_weights_three([shared_ab[0], shared_ac[0], shared_ad[0]])

    print("\nAggregated fc_out weights shape:", agg_weight.shape)

    global_model = create_global_model_agg_weights(agg_bias, agg_weight, fc1_weight, fc1_bias, False)

    print(f"Sequential_data_loaders: {dataloaders_global_model}, {len(dataloaders_global_model)}")

    global_models = [copy.deepcopy(global_model) for i in range(4)]

    optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(global_models)}
    
    
    # global_models = devertifl_global_model(global_models, device, dataloaders_global_model, optimizers, 5, 5, False)
    global_models = devertifl_global_model_with_freeze(global_models, device, dataloaders_global_model, optimizers, 5, 5, False, 5)
    f1_before, acc_before, report_before = evaluate_plain_vertifl(global_models, device, test_loader, False)
    print(f"Accuracy: {acc_before}, F1-Score: {f1_before}")
    print(report_before)

    loss_before = loss_eval(global_models, train_loader)
    confi_forgotten_before, confi_unseen_before = confidence_score_difference(global_models, train_loader, test_loader)
    mia_acc_before = adversarial_mia_attack(global_models, train_loader, test_loader, device)

    remaining_shared_models = [shared_ab, shared_ac, shared_ad]

    new_global_model = forget_shared_model_direct("shared_ad", remaining_shared_models, device, model_scores, retraining_from_scratch)
    if new_global_model is not None:
        new_global_models = [copy.deepcopy(new_global_model) for i in range(4)]
        new_optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(new_global_models)}

        # new_global_models = devertifl_global_model(new_global_models, device, dataloaders_global_model, new_optimizers, 5, 5, True)
        new_global_models = devertifl_global_model_with_freeze(new_global_models, device, dataloaders_global_model, new_optimizers, 5, 2, True, 1)

        print("\nEvaluating Global Model after unlearning 'shared_ab'...")

        f1_after, acc_after, report_after = evaluate_plain_vertifl(new_global_models, device, test_loader, True)

        print("Running Unlearning Verification after Forgetting...")
        loss_after = loss_eval(new_global_models, train_loader, unlearning_step=True)
        confi_forgotten_after, confi_unseen_after = confidence_score_difference(new_global_models, train_loader,
                                                                                test_loader, unlearning_step=True)
        mia_acc_after = adversarial_mia_attack(new_global_models, train_loader, test_loader, device,unlearning_step=True)

        total_time = time.time() - start_time
        cpu_mem = psutil.virtual_memory().used / (1024.0 ** 3)
        print(f"Overall CPU Memory Usage: {cpu_mem:.2f} GB, Total Time: {total_time:.2f} seconds")


        # total_cpu_usage_sm += cpu_fine_tune + cpu
        print(f"Total CPU Usage: {total_cpu_usage_sm} GB")
        if scenario == EXTREME_NON_IID_CASE:
            plot_report_diff(report_before, report_after, dataset_name, EXTREME_NON_IID_CASE, retraining_from_scratch)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     dataset_name,
                                     EXTREME_NON_IID_CASE, retraining_from_scratch)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, EXTREME_NON_IID_CASE)
        elif scenario == NON_IID_CASE:
            plot_report_diff(report_before, report_after, dataset_name, NON_IID_CASE, retraining_from_scratch)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     dataset_name,
                                     NON_IID_CASE, retraining_from_scratch)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, NON_IID_CASE)

        else:
            plot_report_diff(report_before, report_after, dataset_name, IID_CASE, retraining_from_scratch)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     dataset_name,
                                     IID_CASE, retraining_from_scratch)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, NON_IID_CASE)
        return cpu_mem, total_time


def create_global_model_agg_weights(agg_bias, agg_weight, fc1_weight, fc1_bias, rfs):
    global_model = GlobalModel(input_size=784, hidden_size=65, num_classes=10)
    if rfs:
        print("Retraining from scratch, non eed to aggregate classifier weights...")
        return global_model # do not share weights

    # global_model.fc1.weight.data.copy_(fc1_weight)
    # global_model.fc1.bias.data.copy_(fc1_bias)

    global_model.fc_out.weight.data.copy_(agg_weight)
    global_model.fc_out.bias.data.copy_(agg_bias)
    print("Global model fc_out initialized with aggregated shared model fc_out weights.")
    return global_model


def get_shared_models(dataloaders, device, federated_rounds=5):
    shared_ab = [SharedModel(), SharedModel()]  # m1 and m2
    shared_ac = [SharedModel(), SharedModel()]  # m1 and m3
    shared_ad = [SharedModel(), SharedModel()]  # m1 and m4
    total_resources_shared_model_training = 0
    model_scores = []
    print("Training SharedModels Deverti-FL Style")

    for federated_round in range(federated_rounds):
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_ab = executor.submit(
                train_shared_models,
                shared_ab,
                device,
                [dataloaders[0], dataloaders[1]],
                [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ab],
                5,
                True
            )
            future_ac = executor.submit(
                train_shared_models,
                shared_ac,
                device,
                [dataloaders[0], dataloaders[2]],
                [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ac],
                5,
                True
            )
            future_ad = executor.submit(
                train_shared_models,
                shared_ad,
                device,
                [dataloaders[0], dataloaders[3]],
                [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ad],
                5,
                True
            )
            shared_ab, total_resource_ab, ab_acc = future_ab.result()
            shared_ac, total_resource_ac, ac_acc = future_ac.result()
            shared_ad, total_resource_ad, ad_acc = future_ad.result()

        total_resources_shared_model_training += total_resource_ab + total_resource_ac + total_resource_ad
        selective_exchange_gradients(shared_ab, 260)
        selective_exchange_gradients(shared_ac, 260)
        selective_exchange_gradients(shared_ad, 260)
        federated_round_time = time.time() - start_time
        print(f"Federated Rounds [{federated_round + 1}/{federated_rounds}] - Time: {federated_round_time:.2f}s, Resource Consumption: {total_resources_shared_model_training:.4f} GB")
        print(f"Accuracy: {ab_acc}, {ac_acc}, {ad_acc}")
        model_scores = [ab_acc, ac_acc, ad_acc]
    return shared_ab, shared_ac, shared_ad, total_resources_shared_model_training, model_scores

def vertical_partition_rotating_flattened(dataset, num_participants, label_spec=None):

    partitions = [[] for _ in range(num_participants)]
    feature_dim = dataset[0][0].shape[0]  # expected to be 784

    for image, label in dataset:
        # image is a 1D tensor of length 784.
        # Create a zeroed copy for each participant.
        partition_images = [torch.zeros_like(image) for _ in range(num_participants)]
        for i in range(feature_dim):
            participant = i % num_participants
            partition_images[participant][i] = image[i]
        # Append each partition's image, applying label_spec filtering if provided.
        for p in range(num_participants):
            if label_spec is not None and p in label_spec:
                target_label, keep_prob = label_spec[p]
                # If keep_prob==1.0, keep sample only if label equals target_label.
                if keep_prob == 1.0 and label != target_label:
                    continue
                # For probabilistic filtering, if label matches target, include only with probability keep_prob.
                if keep_prob < 1.0 and label == target_label and np.random.rand() > keep_prob:
                    continue
            partitions[p].append((partition_images[p], label))

    # Wrap each partition's list into a TensorDataset.
    dataloaders = []
    for p in range(num_participants):
        data_tensors, labels = zip(*partitions[p])
        partitions[p] = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
        dataloaders.append(DataLoader(partitions[p], batch_size=500, shuffle=True))
    return dataloaders


def get_data_loader(dataset, scenario=None, label_spec_non_iid=None, label_spec_extreme_non_iid=None, fixed_size=392, num_partitions=4):
    num_partitions = 4
    if scenario == EXTREME_NON_IID_CASE and label_spec_extreme_non_iid:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size, label_spec_extreme_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions, label_spec=label_spec_extreme_non_iid)
    elif scenario == NON_IID_CASE and label_spec_non_iid:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size, label_spec_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions, label_spec=label_spec_non_iid)
    else:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size, label_spec_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions, label_spec=label_spec_non_iid)

    return dataloaders

if __name__ == "__main__":
    set_seed(0)
    cpu_mem_mnist = []
    cpu_mem_fashion = []
    total_time_mnist = []
    total_time_fashion = []

    start_time = time.time()
    cpu_mem_iid_mnist, total_time_iid_mnist = dvfu_framework(IID_CASE, MNIST, False)
    cpu_mem_iid_mnist_rtfs, total_time_iid_mnist_rtfs = dvfu_framework(IID_CASE, MNIST, True)
    cpu_mem_mnist.append(cpu_mem_iid_mnist)
    cpu_mem_mnist.append(cpu_mem_iid_mnist_rtfs)
    total_time_mnist.append(total_time_iid_mnist)
    total_time_mnist.append(total_time_iid_mnist_rtfs)
    #
    #
    #
    cpu_mem_non_iid_mnist, total_time_non_iid_mnist = dvfu_framework(NON_IID_CASE, MNIST, False)
    cpu_mem_non_iid_mnist_rtfs, total_time_non_iid_mnist_rtfs = dvfu_framework(NON_IID_CASE, MNIST, True)

    cpu_mem_mnist.append(cpu_mem_non_iid_mnist)
    cpu_mem_mnist.append(cpu_mem_non_iid_mnist_rtfs)
    total_time_mnist.append(total_time_non_iid_mnist)
    total_time_mnist.append(total_time_non_iid_mnist_rtfs)
    #
    #
    cpu_mem_extreme_mnist, total_time_extreme_mnist = dvfu_framework(EXTREME_NON_IID_CASE, MNIST, False)
    cpu_mem_extreme_mnist_rtfs, total_time_extreme_mnist_rtfs = dvfu_framework(EXTREME_NON_IID_CASE, MNIST, True)

    cpu_mem_mnist.append(cpu_mem_extreme_mnist)
    cpu_mem_mnist.append(cpu_mem_extreme_mnist_rtfs)
    total_time_mnist.append(total_time_extreme_mnist)
    total_time_mnist.append(total_time_extreme_mnist_rtfs)



    cpu_mem_iid_fashion, total_time_iid_fashion = dvfu_framework(IID_CASE, FASHION_MNIST, False)
    cpu_mem_iid_fashion_rtfs, total_time_iid_fashion_rtfs = dvfu_framework(IID_CASE, FASHION_MNIST, True)

    cpu_mem_fashion.append(cpu_mem_iid_fashion)
    cpu_mem_fashion.append(cpu_mem_iid_fashion_rtfs)
    total_time_fashion.append(total_time_iid_fashion)
    total_time_fashion.append(total_time_iid_fashion_rtfs)



    cpu_mem_non_iid_fashion, total_time_non_iid_fashion = dvfu_framework(NON_IID_CASE, FASHION_MNIST, False)
    cpu_mem_non_iid_fashion_rtfs, total_time_non_iid_fashion_rtfs = dvfu_framework(NON_IID_CASE, FASHION_MNIST, True)

    cpu_mem_fashion.append(cpu_mem_non_iid_fashion)
    cpu_mem_fashion.append(cpu_mem_non_iid_fashion_rtfs)
    total_time_fashion.append(total_time_non_iid_fashion)
    total_time_fashion.append(total_time_non_iid_fashion_rtfs)

    cpu_mem_extreme_fashion, total_time_extreme_fashion = dvfu_framework(EXTREME_NON_IID_CASE, FASHION_MNIST, False)
    cpu_mem_extreme_fashion_rtfs, total_time_extreme_fashion_rtfs = dvfu_framework(EXTREME_NON_IID_CASE, FASHION_MNIST, True)

    cpu_mem_fashion.append(cpu_mem_extreme_fashion)
    cpu_mem_fashion.append(cpu_mem_extreme_fashion_rtfs)
    total_time_fashion.append(total_time_extreme_fashion)
    total_time_fashion.append(total_time_extreme_fashion_rtfs)
    #
    #
    # resource_usage(cpu_mem_mnist, total_time_mnist, cpu_mem_fashion, total_time_fashion)