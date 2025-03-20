# TODO: compare time perspective of retraining the whole federation vs only the global model
# TODO: Taking average of the hidden outputs of the local models and then feeding it to the global model to make faster
# TODO: accuracy before and after unlearning for DVFU approach vs retrain from scratch?
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import psutil
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
from plotting import plot_unlearning_findings, plot_report_diff

warnings.simplefilter(action='ignore', category=FutureWarning)


########################################### Data Loading ###########################################
# Fashion-MNIST data
def load_fashion_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset


########################################### Constants ###########################################
BATCH_SIZE = 500
PARTIES = 4
non_IID = False
EXTREME_NON_IID_CASE = "extreme non-IID Case"
NON_IID_CASE = "non-IID Case"
IID_CASE = "IID Case"
MNIST = "MNIST"

label_spec_extreme_non_iid = {
    0: (5, 0.0),
    1: (5, 1.0),
    2: (5, 0.0),
    3: (5, 0.0),
}

label_spec_normal_non_iid = {
    0: (5, 0.1),
    1: (5, 0.7),
    2: (5, 0.0),
    3: (5, 0.2),
}

label_spec_even_iid = None

########################################### Model Definitions ###########################################
class LocalModel(nn.Module):
    def __init__(self, input_size=196, hidden_size=260, num_classes=10):
        super(LocalModel, self).__init__()
        self.fc = nn.Linear(hidden_size, num_classes)


class SharedModel(nn.Module):
    def __init__(self, input_size=392, hidden_size=261, num_classes=10):  
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # non-inplace
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_output = self.relu(self.fc1(x))
        classification_output = self.fc_out(hidden_output)
        return hidden_output, classification_output

class GlobalModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=261, num_classes=10):
        super(GlobalModel, self).__init__()
        # Global model takes input from concatenating outputs from n shared models.
        # Its fc1 expects input size = hidden_size * 3.
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
############################################ Tracking Resources ############################################
def track_resource_usage(epoch, phase="Training"):
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    print(f"[{phase} - Epoch {epoch+1}] CPU Memory: {cpu_mem:.2f} GB, GPU Memory: {gpu_mem:.2f} GB")

######################################## Data Preprocessing: Non-IID or IID ########################################
def force_batch_size(tensor, target_size):
    if tensor.size(0) == target_size:
        return tensor
    elif tensor.size(0) > target_size:
        return tensor[:target_size]
    else:
        indices = torch.randint(0, tensor.size(0), (target_size,))
        return tensor[indices]


################################# Unlearning Metrics #################################
def membership_inference_attack(global_model, dataloader, unlearning_step=False):
    global_model.eval()
    with torch.no_grad():
        data, labels = next(iter(dataloader))
        if unlearning_step:
            data = torch.cat((data[:, :196], data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
        else:
            data = data
        logits = global_model(data.float())
        loss = F.cross_entropy(logits, labels)
        print(f"Loss on forgotten client's data: {loss.item():.4f}")
        return loss.item()


def confidence_score_difference(global_model, train_loader, test_loader, unlearning_step=False):
    global_model.eval()
    with torch.no_grad():
        train_data, _ = next(iter(train_loader))
        test_data, _ = next(iter(test_loader))
        
        if unlearning_step:
            train_data = torch.cat((train_data[:, :196], train_data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
            test_data = torch.cat((test_data[:, :196], test_data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
        else:
            train_data = train_data
            test_data = test_data
        client_confidence = F.softmax(global_model(train_data.float()), dim=1).max(dim=1)[0].mean().item()
        unseen_confidence = F.softmax(global_model(test_data.float()), dim=1).max(dim=1)[0].mean().item()

        print(f"Confidence on forgotten client data: {client_confidence:.4f}, Unseen data: {unseen_confidence:.4f}")

        return client_confidence, unseen_confidence


def adversarial_mia_attack(global_model, train_loader, test_loader, device, num_epochs=5, unlearning_step=False):
    global_model.eval()

    labels = torch.cat((torch.ones(BATCH_SIZE), torch.zeros(BATCH_SIZE))).to(device)
    with torch.no_grad():
        client_data, _ = next(iter(train_loader))
        unseen_data, _ = next(iter(test_loader))

        if unlearning_step:
            client_data = torch.cat((client_data[:, :196], client_data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
            unseen_data = torch.cat((unseen_data[:, :196], unseen_data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
        else:
            client_data = client_data
            unseen_data = unseen_data

        client_logits = global_model(client_data.float())
        unseen_logits = global_model(unseen_data.float())
    attack_data = torch.cat((client_logits, unseen_logits), dim=0)
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



# DevertiFL Shared Model Training

def selective_exchange_local_gradients(shared_model):
    """
    Averages the gradients of corresponding parameters between the two local models
    inside a single shared model. This encourages both local models to update
    in a coordinated manner.
    """
    params_m1 = list(shared_model.local_model_m1.parameters())
    params_other = list(shared_model.local_model_other.parameters())
    
    for p1, p2 in zip(params_m1, params_other):
        if p1.grad is not None and p2.grad is not None:
            avg_grad = (p1.grad.data + p2.grad.data) / 2.0
            p1.grad.data.copy_(avg_grad.clone())
            p2.grad.data.copy_(avg_grad.clone())


def train_shared_models_from_locals(shared_models, train_loader, device, num_epochs=5):
    """
    Trains the list of shared models.
    Each input image is split into four segments:
      m1: features [0:196]
      m2: features [196:392]
      m3: features [392:588]
      m4: features [588:784]
    
    The three shared models are trained as:
      - shared_models[0]: receives (m1, m2)
      - shared_models[1]: receives (m1, m3)
      - shared_models[2]: receives (m1, m4)
    
    After backpropagation, gradients are selectively exchanged between the
    two local models in each shared model.
    """
    for model in shared_models:
        model.to(device)
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in shared_models]
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Split the flattened image into four parts
            m1 = images[:, :196]
            m2 = images[:, 196:392]
            m3 = images[:, 392:588]
            m4 = images[:, 588:]
            
            # Forward pass through each shared model:
            output1 = shared_models[0](m1, m2)  # SharedModel1: (m1, m2)
            output2 = shared_models[1](m1, m3)  # SharedModel2: (m1, m3)
            output3 = shared_models[2](m1, m4)  # SharedModel3: (m1, m4)
            
            loss1 = criterion(output1, labels)
            loss2 = criterion(output2, labels)
            loss3 = criterion(output3, labels)
            total_loss = loss1 + loss2 + loss3
            
            # Zero gradients for each optimizer
            for opt in optimizers:
                opt.zero_grad()
            
            # Backward pass
            total_loss.backward()
            
            # Within each shared model, exchange gradients between its two local models.
            for model in shared_models:
                selective_exchange_local_gradients(model)
            
            # Update parameters
            for opt in optimizers:
                opt.step()
                
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Total Loss: {total_loss.item():.4f}")
    
    return shared_models
            

############################################  Knowledge Distillation ############################################
def train_distilled(teacher_model, train_loader, device, temperature=2.0, num_epochs=5):
    student_model = GlobalModel(input_size=588, hidden_size=261, num_classes=10)
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
def aggregate_fc_out_weights(state_dicts):
    weights = [sd['fc_out.weight'] for sd in state_dicts]
    biases = [sd['fc_out.bias'] for sd in state_dicts]
    aggregated_weight = torch.stack(weights).mean(dim=0)
    aggregated_bias = torch.stack(biases).mean(dim=0)
    return aggregated_weight, aggregated_bias


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
def evaluate_model(global_model: GlobalModel, test_loader: DataLoader, device, shared_models=None,
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
                data = torch.cat((data[:, :196], data[:, 392:]), dim=1)  # keep only m1, m3, and m4
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
def fine_tune_on_remaining_data(global_model, train_loader, device, num_epochs=5):
    print(f"Fine-tuning global model on remaining parties")
    cpu_mem_fine_tune = 0
    for param in global_model.fc_out.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    global_model.to(device)
    global_model.train()

    for epoch in range(num_epochs):
        start_time = time.time()
        for inputs, labels in train_loader:
            if inputs.shape[0] == 0:  # Skip batch if empty after filtering
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs

            inputs = torch.cat((inputs[:, :196], inputs[:, 392:]), dim=1)  # Keep only m1, m3, and m4
            optimizer.zero_grad()
            outputs = global_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time
        print(f"Fine-tune Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")
    
    return global_model

############################### forget step with subtraction ###############################
def forget_shared_model_avg(forget_name, shared_models, train_loader, device, num_epochs_global=5):
    # Remove the shared model with name forget_name from the list.
    all_names = ["shared_ab", "shared_ac", "shared_ad"]
    remaining_models = []
    forgotten_state = None
    for name, model in zip(all_names, shared_models):
        if name == forget_name:
            forgotten_state = model.state_dict()
        else:
            remaining_models.append(model)
        
    if forgotten_state is None or len(remaining_models) == 0:
        return None

    orig_state_dicts = [sm.state_dict() for sm in shared_models]
    orig_agg_weight, orig_agg_bias = aggregate_fc_out_weights(orig_state_dicts)
    print("The error is here")
    forgotten_weight = forgotten_state['fc_out.weight']
    forgotten_bias = forgotten_state['fc_out.bias']
    new_agg_weight = orig_agg_weight - forgotten_weight
    new_agg_bias = orig_agg_bias - forgotten_bias
    
    diff = torch.norm(orig_agg_weight[5] - new_agg_weight[5]).item()
    print(f"Norm of difference for label 5 weights: {diff:.4f}")

    
    # Create a new GlobalModel and initialize its fc_out using the new aggregate.
    new_global_model = GlobalModel(input_size=588,hidden_size=261, num_classes=10)
    new_global_model.fc2.weight.data.copy_(new_agg_weight)
    new_global_model.fc2.bias.data.copy_(new_agg_bias)
    print("Global model fc_out updated using direct average of remaining shared models after unlearning", forget_name)
    # finetune is done in separate step
    return new_global_model
        cpu_mem_fine_tune += track_resource_usage(epoch, "Fine-Tuned Global Model Training")
        print(f"Fine-tune Epoch [{epoch + 1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")

    return global_model, cpu_mem_fine_tune


############################### Forget step direct ###############################
def forget_shared_model_direct(forget_name, shared_models, device):
    all_names = ["shared_ab", "shared_ac", "shared_ad"]

    remaining_models = []
    for name, model in zip(all_names, shared_models):
        if name != forget_name:
            remaining_models.append(model)

    print(f"Remaining models after forgetting {forget_name}: {len(remaining_models)}")
    if len(remaining_models) == 0:
        print("No remaining models available. Cannot unlearn.")
        return None

    # aggregate fc_out weights and biases from the remaining models
    remaining_state_dicts = [sm[0].state_dict() for sm in remaining_models]
    new_agg_weight, new_agg_bias = aggregate_fc_out_weights(remaining_state_dicts)
    
    # Create a new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=588, hidden_size=261, num_classes=10).to(device)
    
    # Initialize the global model's classifier layer with the new aggregated weights and biases
    new_global_model.fc2.weight.data.copy_(new_agg_weight)
    new_global_model.fc2.bias.data.copy_(new_agg_bias)
    

    # new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=588, hidden_size=260, num_classes=10).to(device)

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
def dvfu_framework(scenario: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # partition the dataset into 4 disjoint subsets
    dataloaders = get_data_loader(dataset, label_spec_extreme_non_iid, scenario)

    # Initialize three shared models.
    shared_ab, shared_ac, shared_ad, total_cpu_usage_sm = get_shared_models(dataloaders, device)

    print("SharedModels training complete.")
    # compare_state_dicts(shared_ab[0], shared_ab[1])
    # compare_state_dicts(shared_ac[0], shared_ac[1])
    # compare_state_dicts(shared_ad[0], shared_ad[1])    

    state_dicts = [shared_ab[0].state_dict(), shared_ac[0].state_dict(), shared_ad[0].state_dict()]
    agg_weight, agg_bias = aggregate_fc_out_weights(state_dicts)
    print("\nAggregated fc_out weights shape:", agg_weight.shape)

    global_model = create_global_model_agg_weights(agg_bias, agg_weight)

    print("Fine-tuning Global Model on raw %s training data" % MNIST)
    global_model, cpu = train_global_model(global_model, train_loader, device, num_epochs=5)

    print("\nEvaluating Global Model on %s test data..." % MNIST)
    acc_before, f1_before, report_before = evaluate_model(global_model, test_loader, device,
                                                          [shared_ab, shared_ac, shared_ad], unlearning_step=False)

    # print("Running Unlearning Verification before Forgetting...")
    loss_before = membership_inference_attack(global_model, train_loader)
    confi_forgotten_before, confi_unseen_before = confidence_score_difference(global_model, train_loader, test_loader)
    mia_acc_before = adversarial_mia_attack(global_model, train_loader, test_loader, device)

    # Now simulate unlearning: Forget one shared model (e.g. "shared_ab")
    print("\nUnlearning: Removing shared model 'shared_ab'...")

    # Unlearning
    remaining_shared_models = [shared_ab, shared_ac, shared_ad]

    new_global_model = forget_shared_model_direct("shared_ab", remaining_shared_models, device)
    if new_global_model is not None:
        new_global_model, cpu_fine_tune = fine_tune_on_remaining_data(new_global_model, train_loader,
                                                                      device=device, num_epochs=5)

        # global_model = train_distilled(new_global_model, train_loader, device, temperature=2.0, num_epochs=5)

        print("\nEvaluating Global Model after unlearning 'shared_ab'...")
        acc_after, f1_after, report_after = evaluate_model(new_global_model, test_loader, device, None,
                                                           unlearning_step=True)
        print("Running Unlearning Verification after Forgetting...")
        loss_after = membership_inference_attack(new_global_model, train_loader, unlearning_step=True)
        confi_forgotten_after, confi_unseen_after = confidence_score_difference(new_global_model, train_loader,
                                                                                test_loader, unlearning_step=True)
        mia_acc_after = adversarial_mia_attack(new_global_model, train_loader, test_loader, device,
                                               unlearning_step=True)

        total_cpu_usage_sm += cpu_fine_tune + cpu
        print(f"Total CPU Usage: {total_cpu_usage_sm} GB")
        if scenario == EXTREME_NON_IID_CASE:
            plot_report_diff(report_before, report_after, MNIST, EXTREME_NON_IID_CASE)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     MNIST,
                                     EXTREME_NON_IID_CASE)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, EXTREME_NON_IID_CASE)
        elif scenario == NON_IID_CASE:
            plot_report_diff(report_before, report_after, MNIST, NON_IID_CASE)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     MNIST,
                                     NON_IID_CASE)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, NON_IID_CASE)

        else:
            plot_report_diff(report_before, report_after, MNIST, IID_CASE)
            plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after,
                                     confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after,
                                     MNIST,
                                     IID_CASE)
            # plot_resource_comparison(360.287, 51.64, 210.339, 29.06, MNIST, NON_IID_CASE)



def create_global_model_agg_weights(agg_bias, agg_weight):
    global_model = GlobalModel(input_size=784, hidden_size=260, num_classes=10)
    global_model.fc_out.weight.data.copy_(agg_weight)
    global_model.fc_out.bias.data.copy_(agg_bias)
    print("Global model fc_out initialized with aggregated shared model fc_out weights.")
    return global_model


def get_shared_models(dataloaders, device, federated_rounds=5):
    shared_ab = [SharedModel(), SharedModel()]  # m1 and m2
    shared_ac = [SharedModel(), SharedModel()]  # m1 and m3
    shared_ad = [SharedModel(), SharedModel()]  # m1 and m4
    total_resources_shared_model_training = 0
    print("Training SharedModels Deverti-FL Style")

    for federated_round in range(federated_rounds):
        start_time = time.time()
        shared_ab, total_resource_ab = train_shared_models(shared_ab, device, [dataloaders[0], dataloaders[1]],
                                                           [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ab], 5,
                                                           196)
        shared_ac, total_resource_ac = train_shared_models(shared_ac, device, [dataloaders[0], dataloaders[2]],
                                                           [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ac], 5,
                                                           196)
        shared_ad, total_resource_ad = train_shared_models(shared_ad, device, [dataloaders[0], dataloaders[3]],
                                                           [optim.Adam(sm.parameters(), lr=0.001) for sm in shared_ad], 5,
                                                           196)
        total_resources_shared_model_training += total_resource_ab + total_resource_ac + total_resource_ad
        selective_exchange_gradients(shared_ab, 260)
        selective_exchange_gradients(shared_ac, 260)
        selective_exchange_gradients(shared_ad, 260)
        federated_round_time = time.time() - start_time
        print(f"Federated Rounds [{federated_round + 1}/{federated_rounds}] - Time: {federated_round_time:.2f}s, Resource Consumption: {total_resources_shared_model_training:.4f} GB")

    return shared_ab, shared_ac, shared_ad, total_resources_shared_model_training


def get_data_loader(dataset, label_spec_scenario, scenario=None):
    if scenario == EXTREME_NON_IID_CASE:
        dataloaders = partition_dataset(dataset, 4, 500, True, 392, label_spec_scenario)
    elif scenario == NON_IID_CASE:
        dataloaders = partition_dataset(dataset, 4, 500, True, 392, label_spec_scenario)
    else:
        dataloaders = partition_dataset(dataset, 4, 500, True, 392)
    return dataloaders


if __name__ == "__main__":
    start_time = time.time()
    dvfu_framework(IID_CASE)
    total_time = time.time() - start_time
    track_resource_usage(0, "Complete Execution")
    print(f"Total execution time: {total_time:.2f} seconds")
