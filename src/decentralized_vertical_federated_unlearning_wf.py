import concurrent.futures
import copy
import random
import time
import warnings
from typing import Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from plotting import plot_unlearning_findings, plot_report_diff, resource_usage

TARGET_SM = "shared_ad"

warnings.simplefilter(action='ignore', category=FutureWarning)

########################################### Constants ###########################################
MNIST = "MNIST"
FASHION_MNIST = "Fashion-MNIST"
BANK_MARKETING = "Bank Marketing"
BATCH_SIZE = 32
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

# F-MNIST
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

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)
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
def membership_inference_attack(global_models, dataloader,
                                unlearning_step=False):  # TODO: better method name, should be loss_eval
    for gm in global_models:
        gm.eval()
    with torch.no_grad():
        data, labels = next(iter(dataloader))
        if unlearning_step:
            # index % 4 == 3 (Partition 4)
            mask = torch.tensor([(j % 4 != 3) for j in range(784)], device=data.device, dtype=torch.bool)
            data = data * mask
        hidden_outputs = generate_hidden_outputs(global_models, data, data.device)
        combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

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
            mask = torch.tensor([(j % 4 != 3) for j in range(784)], device=device, dtype=torch.bool)
            train_data = train_data * mask
            test_data = test_data * mask

        hidden_client_outs = generate_hidden_outputs(global_models, train_data.float(), device)
        hidden_unseen_outs = generate_hidden_outputs(global_models, test_data.float(), device)

        combined_hidden_client_outputs = torch.cat(hidden_client_outs, dim=1)
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


################################## DevertiFL Shared Model Training (Code from DevertiFL repo) ##################################
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


################################## Data Partitioning ##################################
def partition_dataset_round_robin_order_preserving(dataset, num_partitions, batch_size=500, shuffle=True,
                                                   fixed_size=784, label_spec=None):
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


def partition_dataset_round_robin_order_preserving_optimized(dataset, num_partitions, batch_size=500, shuffle=True,
                                                             fixed_size=784, label_spec=None):
    active_indices = [list(range(i, fixed_size, num_partitions)) for i in range(num_partitions)]

    masks = []
    for i in range(num_partitions):
        mask = torch.zeros(fixed_size, dtype=torch.bool)
        mask[active_indices[i]] = True
        masks.append(mask)

    partitions = [[] for _ in range(num_partitions)]

    for data, label in dataset:
        for i in range(num_partitions):
            if label_spec is not None and i in label_spec:
                target_label, keep_prob = label_spec[i]
                if keep_prob == 1.0 and label != target_label:
                    continue
                elif keep_prob < 1.0 and label == target_label and np.random.rand() > keep_prob:
                    continue
            new_data = torch.where(masks[i].to(data.device), data, torch.zeros_like(data))
            partitions[i].append((new_data, label))

    dataloaders = []
    for part in partitions:
        if part:
            data_tensors, labels = zip(*part)
            dataset_part = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
            loader = DataLoader(dataset_part, batch_size=batch_size, shuffle=shuffle)
            dataloaders.append(loader)
        else:
            print("Warning: One partition is empty and will be skipped")

    return dataloaders

############################### Aggregate fc_1 and fc_out Weights from Shared Models ###############################
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


def aggregate_fc1_weights(state_dicts):
    weights_list = [sd['fc1.weight'] for sd in state_dicts]
    biases_list = [sd['fc1.bias'] for sd in state_dicts]

    # Average across shared models.
    W_shared = torch.stack(weights_list, dim=0).mean(dim=0)  # shape: (130, 392)
    b_shared = torch.stack(biases_list, dim=0).mean(dim=0)  # shape: (130,)

    # Step 2: Reduce the output dimension from 130 to 65 by averaging every two neurons.
    # Reshape (130, 392) to (65, 2, 392) and average over dimension 1.
    W_reduced = W_shared.view(65, 2, 392).mean(dim=1)  # shape: (65, 392)
    b_reduced = b_shared.view(65, 2).mean(dim=1)  # shape: (65,)

    # Step 3: Increase the input dimension from 392 to 784 by duplicating the weights.
    W_global = torch.cat([W_reduced, W_reduced], dim=1)  # shape: (65, 784)

    return W_global, b_reduced


############################### Forget step direct ###############################
def forget_shared_model_direct(forget_name, shared_models, device, model_scores, rtfs):
    print(f"\nUnlearning: Removing shared model '{forget_name}'...")
    all_names = ["shared_ab", "shared_ac", "shared_ad"]

    remaining_models = []
    remaining_models_scores = []
    for name, model, score in zip(all_names, shared_models, model_scores):
        if name != forget_name:
            remaining_models.append(model)
            remaining_models_scores.append(score)

    print(f"Remaining shared models after forgetting {forget_name}: {len(remaining_models)}")
    if len(remaining_models) == 0:
        print("No remaining shared models available. Cannot unlearn.")
        return None

    remaining_state_dicts = [average_state_dicts([sm[0].state_dict(), sm[1].state_dict()]) for sm in remaining_models]

    new_agg_weights_fc_out, new_agg_bias_fc_out = aggregate_fc_out_weights(remaining_state_dicts,
                                                                           remaining_models_scores)
    new_agg_weights_fc1, new_agg_bias_fc1 = aggregate_fc1_weights(remaining_state_dicts)

    # new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=784, hidden_size=65, num_classes=10).to(device)
    if rtfs:
        print("Retraining from scratch, non eed to aggregate classifier weights...")
        return new_global_model

    print("Adding pretrained weights")
    new_global_model.fc1.weight.data.copy_(new_agg_weights_fc1)
    new_global_model.fc1.bias.data.copy_(new_agg_bias_fc1)

    new_global_model.fc_out.weight.data.copy_(new_agg_weights_fc_out)
    new_global_model.fc_out.bias.data.copy_(new_agg_bias_fc_out)

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


###################################### Fine-Tuning Global Models #####################################
def fine_tune_on_remaining_data(models, device, train_loaders, optimizers, epoch, federated_rounds, freeze_rounds=2):
    return fine_tune_gm(models, device, train_loaders, optimizers, epoch, federated_rounds, True, freeze_rounds)


def fine_tune_gm(models, device, train_loaders, optimizers, epoch, federated_rounds,
                 unlearning_step=False, freeze_rounds=2):
    for federated_round in range(1, federated_rounds + 1):
        models = gradual_parameter_freezing(federated_round, freeze_rounds, models)

        print(f"Federated round {federated_round}/{federated_rounds}: "
              f"{'Weights Frozen' if federated_round <= freeze_rounds else 'Weights Unfrozen'}")

        # accumulate losses over the round
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

                    if unlearning_step:
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


def gradual_parameter_freezing(federated_round, freeze_rounds, models):
    for model in models:
        if federated_round <= freeze_rounds:
            # Freeze
            for param in model.fc_out.parameters():
                param.requires_grad = False
            for param in model.fc1.parameters():
                param.require_grad = False
        else:
            # Unfreeze
            for param in model.fc_out.parameters():
                param.requires_grad = True
            for param in model.fc1.parameters():
                param.require_grad = True
    return models


###################################### Evaluation ######################################
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
                mask = torch.tensor([(j % 4 != 3) for j in range(784)], device=data.device, dtype=torch.bool)
                data = data * mask
            hidden_outputs = generate_hidden_outputs(models, data, device)
            combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)

            outputs = torch.zeros(data.size(0), 10, device=device)
            for model in models:
                outputs += model.predict(combined_hidden_outputs)
            outputs /= len(models)

            loss = criterion(outputs, target)
            test_loss += loss.item()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().squeeze().tolist())
            all_targets.extend(target.cpu().tolist())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    report = classification_report(all_targets, all_preds, digits=4, output_dict=True)

    return f1, accuracy, report


###################################### Main Process ######################################
def dvfu_wf_framework(scenario: str, dataset_name: str):
    print(
        f"######################### Scenario: {scenario} for {dataset_name} ######################### ")

    start_time = time.time()
    process = psutil.Process()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_name.upper() == MNIST.upper():
        train_dataset, test_dataset = load_mnist_data()
        label_spec_non_iid, label_spec_extreme_non_iid = label_spec_normal_non_iid_mnist, label_spec_extreme_non_iid_mnist

    elif dataset_name.upper() == FASHION_MNIST.upper():
        train_dataset, test_dataset = load_fashion_mnist_data()
        label_spec_non_iid, label_spec_extreme_non_iid = label_spec_normal_non_iid_fashion_mnist, label_spec_extreme_non_iid_fashion_mnist
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    print(f"Dataset {dataset_name} loaded")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    shared_model_input_size = 392
    global_model_input_size = 784
    num_partitions = 4

    # partition the dataset into x disjoint subsets for both shared models and global model
    print("Partitioning..")
    shared_model_dataloaders = partition_dataset(train_dataset, scenario, label_spec_non_iid,
                                                 label_spec_extreme_non_iid,
                                                 shared_model_input_size, num_partitions)
    global_model_dataloaders = partition_dataset(train_dataset, scenario, label_spec_non_iid,
                                                 label_spec_extreme_non_iid,
                                                 global_model_input_size, num_partitions)
    print("Partitioning complete")

    # trained shared models
    shared_ab, shared_ac, shared_ad, total_cpu_usage_sm, model_scores = get_shared_models(shared_model_dataloaders,
                                                                                          device)

    print("Calculate avg parameters for each shared model")
    avg_shared_ab = average_state_dicts([shared_ab[0].state_dict(), shared_ab[1].state_dict()])
    avg_shared_ac = average_state_dicts([shared_ac[0].state_dict(), shared_ac[1].state_dict()])
    avg_shared_ad = average_state_dicts([shared_ad[0].state_dict(), shared_ad[1].state_dict()])

    state_dicts = [avg_shared_ab, avg_shared_ac, avg_shared_ad]
    fc_out_weights, fc_out_biases = aggregate_fc_out_weights(state_dicts, model_scores)
    fc1_weights, fc1_biases = aggregate_fc1_weights(state_dicts)

    global_model = create_global_model_agg_weights(fc_out_biases, fc_out_weights, fc1_weights, fc1_biases, False)

    global_models = [copy.deepcopy(global_model) for i in range(4)]
    optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(global_models)}

    global_models = fine_tune_gm(global_models, device, global_model_dataloaders, optimizers, 5, 5, False, 1)

    f1_before, acc_before, report_before = evaluate_gm(global_models, device, test_loader, False)
    print(f"Accuracy: {acc_before}, F1-Score: {f1_before}")
    print(report_before)

    loss_before = membership_inference_attack(global_models, train_loader)
    confi_forgotten_before, confi_unseen_before = confidence_score_difference(global_models, train_loader, test_loader)
    mia_acc_before = adversarial_mia_attack(global_models, train_loader, test_loader, device)

    all_shared_models = [shared_ab, shared_ac, shared_ad]

    retraining_from_scratch = False
    new_global_model = forget_shared_model_direct(TARGET_SM, all_shared_models, device, model_scores,
                                                  retraining_from_scratch)

    new_global_models = [copy.deepcopy(new_global_model) for i in range(4)]
    new_optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(new_global_models)}

    freezed_rounds = 1 if scenario == EXTREME_NON_IID_CASE else 0  # 0 for extreme nonIID to better fit the data

    new_global_models = fine_tune_on_remaining_data(new_global_models, device, global_model_dataloaders,
                                                    new_optimizers, 5, 2, freezed_rounds)

    print("\nEvaluating Global Model after unlearning '%s'" % TARGET_SM)

    f1_after, acc_after, report_after = evaluate_gm(new_global_models, device, test_loader, True)

    print("Running Unlearning Verification after Forgetting...")
    loss_after = membership_inference_attack(new_global_models, train_loader, unlearning_step=True)
    confi_forgotten_after, confi_unseen_after = confidence_score_difference(new_global_models, train_loader,
                                                                            test_loader, unlearning_step=True)
    mia_acc_after = adversarial_mia_attack(new_global_models, train_loader, test_loader, device,
                                           unlearning_step=True)

    total_time = time.time() - start_time
    cpu_mem = process.memory_info().rss / (1024.0 ** 3)
    print(f"Overall Framework consumption metrics: \nCPU Memory Usage: {cpu_mem:.2f} GB, Total Time: {total_time:.2f} seconds")
    cpu_mem_wf, total_time_wf = cpu_mem, total_time

    if scenario == EXTREME_NON_IID_CASE:
        plot_report_diff(report_before, report_after, dataset_name, EXTREME_NON_IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 EXTREME_NON_IID_CASE, retraining_from_scratch)
    elif scenario == NON_IID_CASE:
        plot_report_diff(report_before, report_after, dataset_name, NON_IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 NON_IID_CASE, retraining_from_scratch)

    else:
        plot_report_diff(report_before, report_after, dataset_name, IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 IID_CASE, retraining_from_scratch)

    ######################################### Retraining from Scracth #########################################
    print("Retraining from Scratch..")
    global_model = GlobalModel(input_size=784, hidden_size=65, num_classes=10)
    global_models = [copy.deepcopy(global_model) for i in range(4)]
    optimizers = {i: optim.Adam(model.parameters(), lr=0.001) for i, model in enumerate(global_models)}

    global_models = fine_tune_on_remaining_data(global_models, device, global_model_dataloaders, optimizers, 5,
                                                5,5)
    f1_after, acc_after, report_after = evaluate_gm(global_models, device, test_loader, True)
    print(f"Accuracy: {acc_before}, F1-Score: {f1_before}")
    print(report_before)

    loss_after = membership_inference_attack(global_models, train_loader)
    confi_forgotten_after, confi_unseen_after = confidence_score_difference(global_models, train_loader, test_loader)
    mia_acc_after = adversarial_mia_attack(global_models, train_loader, test_loader, device)

    total_time = time.time() - start_time

    cpu_mem = process.memory_info().rss / (1024.0 ** 3)
    cpu_mem_rtfs, total_time_rtfs = cpu_mem, total_time
    retraining_from_scratch = True
    print(f"Overall CPU Memory Usage: {cpu_mem:.2f} MB, Total Time: {total_time:.2f} seconds")

    if scenario == EXTREME_NON_IID_CASE:
        plot_report_diff(report_before, report_after, dataset_name, EXTREME_NON_IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 EXTREME_NON_IID_CASE, retraining_from_scratch)
    elif scenario == NON_IID_CASE:
        plot_report_diff(report_before, report_after, dataset_name, NON_IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 NON_IID_CASE, retraining_from_scratch)

    else:
        plot_report_diff(report_before, report_after, dataset_name, IID_CASE, retraining_from_scratch)
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                 confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                 dataset_name,
                                 IID_CASE, retraining_from_scratch)

    return cpu_mem_wf, total_time_wf, cpu_mem_rtfs, total_time_rtfs


def create_global_model_agg_weights(agg_bias, agg_weight, fc1_weight, fc1_bias, rtfs):
    global_model = GlobalModel(input_size=784, hidden_size=65, num_classes=10)
    if rtfs:
        print("Retraining from scratch, non need to aggregate classifier weights")
        return global_model  # do not use shared model parameters

    global_model.fc1.weight.data.copy_(fc1_weight)
    global_model.fc1.bias.data.copy_(fc1_bias)

    global_model.fc_out.weight.data.copy_(agg_weight)
    global_model.fc_out.bias.data.copy_(agg_bias)
    print("Global model initialized with aggregated shared model params")
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
        print(
            f"Federated Rounds [{federated_round + 1}/{federated_rounds}] - Time: {federated_round_time:.2f}s, Resource Consumption: {total_resources_shared_model_training:.4f} GB")
        print(f"Accuracy: {ab_acc}, {ac_acc}, {ad_acc}")
        model_scores = [ab_acc, ac_acc, ad_acc]
    print("SharedModels training complete")
    return shared_ab, shared_ac, shared_ad, total_resources_shared_model_training, model_scores


def vertical_partition_rotating_flattened(dataset, num_participants, label_spec=None):
    partitions = [[] for _ in range(num_participants)]
    feature_dim = dataset[0][0].shape[0]  # expected to be 784

    for image, label in dataset:
        partition_images = [torch.zeros_like(image) for _ in range(num_participants)]
        for i in range(feature_dim):
            participant = i % num_participants
            partition_images[participant][i] = image[i]
        for p in range(num_participants):
            if label_spec is not None and p in label_spec:
                target_label, keep_prob = label_spec[p]
                # keep_prob==1.0, keep sample only if label equals target_label.
                if keep_prob == 1.0 and label != target_label:
                    continue
                if keep_prob < 1.0 and label == target_label and np.random.rand() > keep_prob:
                    continue
            partitions[p].append((partition_images[p], label))

    dataloaders = []
    for p in range(num_participants):
        data_tensors, labels = zip(*partitions[p])
        partitions[p] = TensorDataset(torch.stack(data_tensors), torch.tensor(labels))
        dataloaders.append(DataLoader(partitions[p], batch_size=500, shuffle=True))
    return dataloaders


def partition_dataset(dataset, scenario=None, label_spec_non_iid=None, label_spec_extreme_non_iid=None, fixed_size=392,
                      num_partitions=4):
    if scenario == EXTREME_NON_IID_CASE and label_spec_extreme_non_iid:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size,
                                                                         label_spec_extreme_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions,
                                                                label_spec=label_spec_extreme_non_iid)
    elif scenario == NON_IID_CASE and label_spec_non_iid:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size,
                                                                         label_spec_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions, label_spec=label_spec_non_iid)
    else:
        if fixed_size == 392:
            dataloaders = partition_dataset_round_robin_order_preserving(dataset, num_partitions, 500, True, fixed_size,
                                                                         label_spec_non_iid)
        elif fixed_size == 784:
            dataloaders = vertical_partition_rotating_flattened(dataset, num_partitions, label_spec=label_spec_non_iid)
        else:
            raise NotImplementedError("FixedSize must be 392 or 784")

    return dataloaders


if __name__ == "__main__":
    set_seed(0)
    start_time = time.time()
    scenarios = [IID_CASE, NON_IID_CASE, EXTREME_NON_IID_CASE]
    bench_mark_datasets = [MNIST, FASHION_MNIST]

    for dataset_name in bench_mark_datasets:
        total_cpu_wf = []
        total_cpu_rtfs = []

        total_time_wf = []
        total_time_rtfs = []
        for scenario in scenarios:
            cpu_mem_dvfu_wf, time_dvfu_wf, cpu_mem_rtfs, time_rtfs = dvfu_wf_framework(scenario, dataset_name)
            total_cpu_wf.append(cpu_mem_dvfu_wf)
            total_cpu_rtfs.append(cpu_mem_rtfs)
            total_time_wf.append(time_dvfu_wf)
            total_time_rtfs.append(time_rtfs)
            print(f"{dataset_name} - {scenario}")
            print(f"DVFU-WF: CPU={cpu_mem_dvfu_wf} and time={time_dvfu_wf}")
            print(f"RTFS: CPU={cpu_mem_rtfs} and time={time_rtfs}")
        resource_usage(total_cpu_wf, total_cpu_rtfs, total_time_wf, total_time_rtfs, dataset_name)

