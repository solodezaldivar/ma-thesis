# import glob
# from tracemalloc import start
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# from torchvision import datasets, transforms
# import os
# import psutil
# import time
# import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import f1_score
# from sklearn.metrics import classification_report
# from hmac import new
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# # Shared model from which we will use the hidden outputs
# class SharedModel(nn.Module):
#     def __init__(self, input_size=392, hidden_size=261, num_classes=10):  
#         super(SharedModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc_out = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         hidden_output = self.relu(self.fc1(x)) 
#         classification_output = self.fc_out(hidden_output)
#         return hidden_output, classification_output

# # Global Model which will has the input size of x times hidden outputs 
# class GlobalModel(nn.Module):
#     def __init__(self, hidden_size=261, num_classes=10):
#         super(GlobalModel, self).__init__()
#         input_size = hidden_size * 3 #<- 3 because in this case we have 3 shared models, maybe change to a var 
#         self.fc1 = nn.Linear(input_size, 512)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, h1, h2, h3):
#         x = torch.cat((h1, h2, h3), dim=1)
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)
    
# class ParticipantModel(nn.Module):
#     def __init__(self, hidden_size=261, num_classes=10):
#         super(ParticipantModel, self).__init__()
#         input_size = hidden_size * 3
#         self.input_layer = nn.Linear(input_size, hidden_size)
#         self.hidden_layer = nn.ReLU()
#         self.output_layer = nn.Linear(input_size, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, h1, h2, h3):
#         x = torch.cat((h1, h2, h3), dim=1)
#         hidden_output = self.hidden_layer(self.input_layer(x))
#         hidden_output = self.dropout(hidden_output)
#         return hidden_output

#     def predict(self, combined_hidden_outputs):
#         output = self.output_layer(combined_hidden_outputs)
#         return output
    
# def track_resource_usage(epoch, phase="Training"):
#     """Prints resource usage stats"""
#     cpu_mem = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
#     gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0  # Convert to GB
#     print(f"[{phase} - Epoch {epoch+1}] CPU Memory Used: {cpu_mem:.2f} GB, GPU Memory Used: {gpu_mem:.2f} GB")


# def split_into_local_models(mnist_data, labels, nonIID=False):
#     if nonIID:
#         return non_IID_partition_data(mnist_data, labels)
#     m1 = mnist_data[:, :196]  
#     m2 = mnist_data[:, 196:392]  
#     m3 = mnist_data[:, 392:588]  
#     m4 = mnist_data[:, 588:]  
#     # print(f"m1: {m1.shape}, m2: {m2.shape}, m3: {m3.shape}, m4: {m4.shape}")

#     return m1, m2, m3, m4

# from collections import Counter

# def non_IID_partition_data(data, labels, batch_size=64):
#     seven_indices = (labels == 7).nonzero(as_tuple=True)[0]
#     non_seven_indices = (labels != 7).nonzero(as_tuple=True)[0]

#     # Extract data partitions
#     m2 = data[seven_indices, 196:392]  # Only '7' samples
#     m1 = data[non_seven_indices, :196]
#     m3 = data[non_seven_indices, 392:588]
#     m4 = data[non_seven_indices, 588:]

#     # Extract corresponding labels for each partition
#     labels_m1 = labels[non_seven_indices]
#     labels_m2 = labels[seven_indices]
#     labels_m3 = labels[non_seven_indices]
#     labels_m4 = labels[non_seven_indices]

#     # Print label distributions
#     print("\n🔹 Label Distributions in Each Partition:")
#     print(f"m1: {dict(Counter(labels_m1.tolist()))}")
#     print(f"m2 (Only 7s): {dict(Counter(labels_m2.tolist()))}")  # Should contain only {7: count}
#     print(f"m3: {dict(Counter(labels_m3.tolist()))}")
#     print(f"m4: {dict(Counter(labels_m4.tolist()))}")

#     # Ensure batch size alignment
#     min_samples = min(m1.shape[0], m3.shape[0], m4.shape[0])

#     def resample_to_match(tensor, target_size):
#         """Resample tensor to match the target size."""
#         indices = torch.randint(0, tensor.shape[0], (target_size,))
#         return tensor[indices]

#     new_size = (min_samples // batch_size) * batch_size  # Make divisible by batch size
#     m1, m3, m4 = m1[:new_size], m3[:new_size], m4[:new_size]
#     m2 = resample_to_match(m2, new_size)

#     print(f"Final Sizes -> m1: {m1.shape}, m2: {m2.shape}, m3: {m3.shape}, m4: {m4.shape}")
#     return m1, m2, m3, m4


# def train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5):
#     shared_ab.to(device)
#     shared_ac.to(device)
#     shared_ad.to(device)

#     optimizer_ab = optim.Adam(shared_ab.parameters(), lr=0.001)
#     optimizer_ac = optim.Adam(shared_ac.parameters(), lr=0.001)
#     optimizer_ad = optim.Adam(shared_ad.parameters(), lr=0.001)
    
#     criterion = nn.CrossEntropyLoss()

#     hidden_outputs_ab = []
#     hidden_outputs_ac = []
#     hidden_outputs_ad = []
#     labels_list = []

#     for epoch in range(num_epochs):
#         start_time = time.time()
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#             IID=True
#             m1, m2, m3, m4 = split_into_local_models(images, labels, True)

#             pad_size = m1.shape[0] - m2.shape[0]
#             m2 = torch.cat((m2, torch.zeros((pad_size, m2.shape[1]), device=m2.device)), dim=0)  # Pad with zeros
#             input_ab = torch.cat((m1, m2), dim=1).to(device)
            
#             input_ac = torch.cat((m1, m3), dim=1).to(device)
#             input_ad = torch.cat((m1, m4), dim=1).to(device)

#             h_ab, pred_ab = shared_ab(input_ab)  
#             h_ac, pred_ac = shared_ac(input_ac)  
#             h_ad, pred_ad = shared_ad(input_ad)  

#             loss_ab = criterion(pred_ab, labels)
#             loss_ac = criterion(pred_ac, labels)
#             loss_ad = criterion(pred_ad, labels)

#             optimizer_ab.zero_grad()
#             loss_ab.backward()
#             optimizer_ab.step()

#             optimizer_ac.zero_grad()
#             loss_ac.backward()
#             optimizer_ac.step()

#             optimizer_ad.zero_grad()
#             loss_ad.backward()
#             optimizer_ad.step()

#             # store hidden outputs for the global model
#             hidden_outputs_ab.append(h_ab.cpu())
#             hidden_outputs_ac.append(h_ac.cpu())
#             hidden_outputs_ad.append(h_ad.cpu())
#             labels_list.append(labels.cpu())

#         epoch_time = time.time() - start_time
#         track_resource_usage(epoch, "Shared Model Training")
#         print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Losses: {loss_ab.item():.4f}, {loss_ac.item():.4f}, {loss_ad.item():.4f}")


#     # export hidden outs
#     torch.save(torch.cat(hidden_outputs_ab, dim=0), "./resources/hidden_outputs/hidden_outputs_ab.pt")
#     torch.save(torch.cat(hidden_outputs_ac, dim=0), "./resources/hidden_outputs/hidden_outputs_ac.pt")
#     torch.save(torch.cat(hidden_outputs_ad, dim=0), "./resources/hidden_outputs/hidden_outputs_ad.pt")
#     torch.save(torch.cat(labels_list, dim=0), "labels.pt")

#     print("Hidden outputs and labels saved!")

# def load_and_split_data(test_ratio=0.2):
#     num_samples = 5000
#     # Load only available hidden representations
#     hidden_outputs = []
#     if os.path.exists("./resources/hidden_outputs/hidden_outputs_ab.pt"):
#         hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ab.pt"))
#     if os.path.exists("./resources/hidden_outputs/hidden_outputs_ac.pt"):
#         hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ac.pt"))
#     if os.path.exists("./resources/hidden_outputs/hidden_outputs_ad.pt"):
#         hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ad.pt"))

#     if not hidden_outputs:
#         raise ValueError("No valid hidden representations found. Did you forget all clients?")

#     # Concatenate only the existing hidden outputs
#     for output in hidden_outputs:
#         output = output[:num_samples]
    
#     hidden_inputs = torch.cat(hidden_outputs, dim=1)
#     hidden_inputs = hidden_inputs[:num_samples]
#     print(f"Hidden Inputs Shape: {hidden_inputs.shape}")

#     # Load labels
#     labels = torch.load("labels.pt") if os.path.exists("labels.pt") else None
#     if labels is None:
#         raise ValueError("Labels file is missing.")

#     labels = labels[:num_samples]
    
#     # Determine split index
#     total_samples = hidden_inputs.size(0)
#     test_size = int(total_samples * test_ratio)
#     train_size = total_samples - test_size

#     # Split data
#     train_data = hidden_inputs[:train_size]
#     test_data = hidden_inputs[train_size:]
#     train_labels = labels[:train_size]
#     test_labels = labels[train_size:]

#     return train_data, test_data, train_labels, test_labels


# def train_global_model(global_model, train_data, train_labels, device, num_epochs=5):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(global_model.parameters(), lr=0.001)

#     train_dataset = TensorDataset(train_data, train_labels)
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #try batch 512 later
    
#     global_model.to(device)
#     global_model.train()
#     for epoch in range(num_epochs):
#         start_time = time.time()
#         print(f"Epoch {epoch+1}/{num_epochs} started at {start_time}")
        
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             num_sources = inputs.shape[1] // 261
#             input_slices = [inputs[:, i*261:(i+1)*261] for i in range(num_sources)]
            
#             # outputs = global_model(inputs[:, :261], inputs[:, 261:512], inputs[:, 512:])
#             outputs = global_model(*input_slices)
#             loss = criterion(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         epoch_time = time.time() - start_time
#         track_resource_usage(epoch, "Global Model Training")   
#         print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")



# def evaluate_global_model(global_model, test_data, test_labels, device):
#     global_model.eval()
#     test_data, test_labels = test_data.to(device), test_labels.to(device)

#     all_preds, all_targets = [], []

#     with torch.no_grad():
#         outputs = global_model(test_data[:, :256], test_data[:, 256:512], test_data[:, 512:])
#         _, predicted = torch.max(outputs, 1)
        
#         all_preds.extend(predicted.cpu().numpy())
#         all_targets.extend(test_labels.cpu().numpy())

#     f1 = f1_score(all_targets, all_preds, average="macro")  # Macro-F1 for multi-class
    
#     accuracy = 100 * (predicted == test_labels).sum().item() / test_labels.size(0)
    
#     print(f'Accuracy on test data: {accuracy:.2f}%')
#     print(f'F1 Score: {f1:.4f}')  # Print F1 Score
    
#     report = classification_report(all_targets, all_preds, digits=4)
#     print(f"\nClassification Report:\n{report}")

#     return accuracy, f1

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load MNIST dataset
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
#     ])
#     train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
#     train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

#     shared_ab = SharedModel() # either use devertiFL or just straight up init the shared model
#     shared_ac = SharedModel()
#     shared_ad = SharedModel()

#     if not os.path.exists("hidden_outputs_ab.pt"):
#         print("Training Shared Models and Storing Hidden Outputs...")
#         train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5)

#     train_data, test_data, train_labels, test_labels = load_and_split_data()

#     global_model = GlobalModel()
#     print("Training Global Model...")
#     train_global_model(global_model, train_data, train_labels, device, num_epochs=5)

#     print("Evaluating Global Model...")
#     evaluate_global_model(global_model, test_data, test_labels, device)

    

# def forget_client(client_name):
#     print(f"Forgetting {client_name} and its dataset contribution...")

#     # Determine which files to remove
#     if client_name == "shared_ab":
#         remove_file = "./resources/hidden_outputs/hidden_outputs_ab.pt"
#     elif client_name == "shared_ac":
#         remove_file = "./resources/hidden_outputs/hidden_outputs_ac.pt"
#     elif client_name == "shared_ad":
#         remove_file = "./resources/hidden_outputs/hidden_outputs_ad.pt"
#     else:
#         print("Invalid client name! Choose from: 'shared_ab', 'shared_ac', 'shared_ad'")
#         return

#     # Delete the specified shared model's hidden representations
#     if os.path.exists(remove_file):
#         os.remove(remove_file)
#         print(f"Removed {remove_file}")
#     else:
#         print(f"{remove_file} does not exist.")

#     # Reload and split the dataset with the remaining features
#     train_data, test_data, train_labels, test_labels = load_and_split_data()

#     # Adjust the global model input size (reduce features)
#     feature_count = train_data.shape[1] // 261
#     print(f"Updating Global Model to accept {feature_count} feature sources instead of 3.")

#     class UpdatedGlobalModel(nn.Module):
#         def __init__(self, hidden_size=261, num_classes=10, num_sources=feature_count):
#             super(UpdatedGlobalModel, self).__init__()
#             input_size = hidden_size * num_sources
#             self.fc1 = nn.Linear(input_size, 512)
#             self.relu = nn.ReLU()
#             self.fc2 = nn.Linear(512, num_classes)

#         def forward(self, *hidden_inputs):
#             x = torch.cat(hidden_inputs, dim=1)
#             x = self.relu(self.fc1(x))
#             return self.fc2(x)

#     # Train the new global model with fewer inputs
#     global_model = UpdatedGlobalModel(num_sources=feature_count)
#     print("\n🚀 Retraining Global Model with fewer features...")
#     train_global_model(global_model, train_data, train_labels, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     evaluate_global_model(global_model, test_data, test_labels, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#     print("Forgetting process complete. Global model updated!")
    

# def evaluate_global_model_per_class(global_model, local_models, data, labels, device, batch_size=64):
#     """
#     Evaluates the global model on the test set and prints the classification report,
#     which includes per-class precision, recall, and F1-score.
#     """
#     global_model.eval()
#     dataset = TensorDataset(data, labels)
#     loader = DataLoader(dataset, batch_size=batch_size)
    
#     all_preds = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch_data, batch_labels in loader:
#             batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
#             # Dynamically split the features based on local models' input sizes:
#             d1 = local_models[0].fc.in_features
#             d2 = local_models[1].fc.in_features
#             d3 = local_models[2].fc.in_features
#             batch_client1 = batch_data[:, :d1]
#             batch_client2 = batch_data[:, d1:d1+d2]
#             batch_client3 = batch_data[:, d1+d2:]
            
#             hidden_outputs = []
#             for model, client_batch in zip(local_models, [batch_client1, batch_client2, batch_client3]):
#                 hidden = model(client_batch)[0]
#                 hidden_outputs.append(hidden)
            
#             outputs = global_model(*hidden_outputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_targets.extend(batch_labels.cpu().numpy())
    
#     print("\nClassification Report:")
#     report = classification_report(all_targets, all_preds, digits=4)
#     print(report)
    
#     # Additionally, return the per-class report as a dictionary if further processing is needed.
#     return report

# # Example usage:
# # Before forgetting m2, evaluate global model with all clients.
# print("Evaluation with all clients (including m2):")

# # After forgetting m2, you would reinitialize local_models (without m2) and retrain the global model.
# # Then, call evaluate_global_model_per_class again on the retrained global model.


# # TODO: compare time perspective of retraining the whole federation vs only the global model
# # TODO: log usage of resources
# # TODO: Batch size to make global model learning faster?
# # TOOD: Taking average of the hidden outputs of the local models and then feeding it to the global model to make faster
# if __name__ == "__main__":
#     start_time = time.time()
#     main()
#     forget_client("shared_ab")
#     complete_time_with_removing = time.time() - start_time
#     track_resource_usage(1, "Complete DevertiFL")
#     print(f"Complete DevertiFL Execution Time with only retraining global model: {complete_time_with_removing:.2f}s")



####################### NONIID PARTITIONING ############################
from cProfile import label
from cgi import test
import glob
from pickletools import optimize
from pydoc import cli
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
IID = True


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

def non_IID_partition_data_extreme(data, labels, batch_size=BATCH_SIZE):
    number_to_forget = 5
    seven_indices = (labels == number_to_forget).nonzero(as_tuple=True)[0]
    non_seven_indices = (labels != number_to_forget).nonzero(as_tuple=True)[0]

    # Extract data partitions
    m2 = force_batch_size(data[seven_indices, 196:392], batch_size)  # Only '5' TODO -> 80% are 5 and 20% are all the other numbers (try with a less extreme scenario where 50% are 5 and 50% are all the other numbers)
    m1 = force_batch_size(data[non_seven_indices, :196], batch_size)
    m3 = force_batch_size(data[non_seven_indices, 392:588], batch_size)
    m4 = force_batch_size(data[non_seven_indices, 588:], batch_size)
    
    return m1, m2, m3, m4, labels

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
    
    attack_model = nn.Sequential(nn.Linear(attack_data.size(1), 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()).to(device)
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


############################################ Split into Local Models ############################################
def split_into_local_models(mnist_data, labels, nonIID=False):
    if nonIID:
       return non_IID_partition_data_extreme(mnist_data, labels)
    m1 = mnist_data[:, :196]  
    m2 = mnist_data[:, 196:392]  
    m3 = mnist_data[:, 392:588]  
    m4 = mnist_data[:, 588:]
    return m1, m2, m3, m4, labels


######################################## Training Utilities for Shared Models ########################################
def train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5, use_nonIID=False):
    shared_ab.to(device)
    shared_ac.to(device)
    shared_ad.to(device)
    optimizer_ab = optim.Adam(shared_ab.parameters(), lr=0.001)
    optimizer_ac = optim.Adam(shared_ac.parameters(), lr=0.001)
    optimizer_ad = optim.Adam(shared_ad.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # TODO: VertiFL for the shared models
    for epoch in range(num_epochs):
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            m1, m2, m3, m4, labels = split_into_local_models(images, labels, use_nonIID)
            labels = labels[:BATCH_SIZE]
        
            input_ab = torch.cat((m1, m2), dim=1).to(device)
            input_ac = torch.cat((m1, m3), dim=1).to(device)
            input_ad = torch.cat((m1, m4), dim=1).to(device)
            
            _, pred_ab = shared_ab(input_ab)
            _, pred_ac = shared_ac(input_ac)
            _, pred_ad = shared_ad(input_ad)
            
            loss_ab = criterion(pred_ab, labels)
            loss_ac = criterion(pred_ac, labels)
            loss_ad = criterion(pred_ad, labels)
            
            optimizer_ab.zero_grad()
            loss_ab.backward()
            optimizer_ab.step()
            optimizer_ac.zero_grad()
            loss_ac.backward()
            optimizer_ac.step()
            optimizer_ad.zero_grad()
            loss_ad.backward()
            optimizer_ad.step()
        
        epoch_time = time.time() - start_time
        track_resource_usage(epoch, "Shared Model Training")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Losses: {loss_ab.item():.4f}, {loss_ac.item():.4f}, {loss_ad.item():.4f}")
    return shared_ab, shared_ac, shared_ad



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
    biases  = [sd['fc_out.bias'] for sd in state_dicts]
    aggregated_weight = torch.stack(weights).mean(dim=0)
    aggregated_bias = torch.stack(biases).mean(dim=0)
    return aggregated_weight, aggregated_bias

###################################### Fine-Tuning Global Model on Raw MNIST Data ######################################
def train_global_model(global_model, train_loader, device, num_epochs=1):
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
        track_resource_usage(epoch, "Global Model Training")
        print(f"Global Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")
    return global_model

######################################### Evaluation #########################################
def evaluate_model(global_model, test_loader, device, shared_models=None, unlearning_step=False):
    global_model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # sm_raw_out = None
            # m1, m2, m3, m4 = split_into_local_models(data, labels)
            # for i, sm in enumerate(shared_models):
            #     if sm_raw_out is None:
            #         sm_raw_out = sm(torch.cat((m1, m2), dim=1))[0].clone()
            #         print("sm_raw_out.shape:", sm_raw_out.shape)
            #     elif i == 1:
            #         inp = torch.cat((m1, m3), dim=1)
            #         sm_raw_out = torch.cat((sm_raw_out, sm(inp)[0].clone()), dim=1)
            #     elif i == 2:
            #         inp = torch.cat((m1, m4), dim=1)
            #         sm_raw_out = torch.cat((sm_raw_out, sm(inp)[0].clone()), dim=1)
            if not unlearning_step:
                outputs = global_model(data)
            else:
                data = torch.cat((data[:, :196], data[:, 392:]), dim=1)  # Keep only m1, m3, and m4
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
def fine_tune_on_remaining_data(global_model, train_loader, forgotten_labels, device, num_epochs=5):
    print(f"Fine-tuning global model on remaining clients, excluding labels: {forgotten_labels}")
    
    for param in global_model.fc2.parameters():
        param.requires_grad = False
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    global_model.to(device)
    global_model.train()
    
    for epoch in range(num_epochs):
        start_time = time.time()
        for inputs, labels in train_loader:
            # Exclude data corresponding to the forgotten labels
            mask = ~torch.isin(labels, torch.tensor(forgotten_labels).to(device))
            inputs, labels = inputs[mask], labels[mask]

            if inputs.shape[0] == 0:  # Skip batch if empty after filtering
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs
            # Remove forgotten feature m2 (196:392)
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


############################### Forget step direct ###############################
def forget_shared_model_direct(forget_name, shared_models, device):
    # Define the names of the shared models
    all_names = ["shared_ab", "shared_ac", "shared_ad"]
    
    # Select only the remaining models
    remaining_models = []
    for name, model in zip(all_names, shared_models):
        if name != forget_name:
            remaining_models.append(model)
    
    if len(remaining_models) == 0:
        print("No remaining models available. Cannot unlearn.")
        return None

    # Aggregate the fc_out weights and biases from the remaining models
    remaining_state_dicts = [sm.state_dict() for sm in remaining_models]
    new_agg_weight, new_agg_bias = aggregate_fc_out_weights(remaining_state_dicts)
    
    # Create a new GlobalModel with updated input size
    new_global_model = GlobalModel(input_size=588, hidden_size=261, num_classes=10).to(device)
    
    # Initialize the global model's classifier layer with the new aggregated weights and biases
    new_global_model.fc2.weight.data.copy_(new_agg_weight)
    new_global_model.fc2.bias.data.copy_(new_agg_bias)
    
    print("Global model fc_out updated using direct average of remaining shared models after unlearning", forget_name)
    return new_global_model

############################################ Main Process ############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    # train_dataset = datasets.FMNIST(root='./data', train=True, transform=transform, download=True)
    # test_dataset  = datasets.FMNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize three shared models.
    shared_ab = SharedModel()  # m1 and m2
    shared_ac = SharedModel()  # m1 and m3
    shared_ad = SharedModel()  # m1 and m4

    print("Training SharedModels")
    shared_ab, shared_ac, shared_ad = train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5, use_nonIID=IID)
    
    # Aggregate the fc_out weights from the shared models.
    state_dicts = [shared_ab.state_dict(), shared_ac.state_dict(), shared_ad.state_dict()]
    agg_weight, agg_bias = aggregate_fc_out_weights(state_dicts)
    print("\nAggregated fc_out weights shape:", agg_weight.shape)
    
    global_model = GlobalModel(input_size=784, hidden_size=261, num_classes=10)
    # Initialize global model's fc2 (classifier) with the aggregated fc_out weights.
    global_model.fc2.weight.data.copy_(agg_weight)
    global_model.fc2.bias.data.copy_(agg_bias)
    print("Global model fc_out initialized with aggregated shared model fc_out weights.")
    
    # Fine-tune GlobalModel on raw MNIST training data.
    print("\nFine-tuning Global Model on raw MNIST training data...")
    train_global_model(global_model, train_loader, device, num_epochs=5)
    
    # Evaluate GlobalModel on test data.
    print("\nEvaluating Global Model on MNIST test data...")
    acc_before, f1_before, report_before = evaluate_model(global_model, test_loader, device, [shared_ab, shared_ac, shared_ad], unlearning_step=False)
    
    # print("Running Unlearning Verification before Forgetting...")
    loss_before = membership_inference_attack(global_model, train_loader)
    confi_forgotten_before, confi_unseen_before = confidence_score_difference(global_model, train_loader, test_loader)
    mia_acc_before = adversarial_mia_attack(global_model, train_loader, test_loader, device)
    
    
    # Now simulate unlearning: Forget one shared model (e.g. "shared_ab")
    print("\nUnlearning: Removing shared model 'shared_ab'...")
    remaining_shared_models = [shared_ab, shared_ac, shared_ad]
    
    
    # In our simulation, remove the one named "shared_ab"
    # new_global_model = forget_shared_model_avg("shared_ab", remaining_shared_models, train_loader, device, num_epochs_global=5)
    
    new_global_model = forget_shared_model_direct("shared_ab", remaining_shared_models, device)
    if new_global_model is not None:
        # knowledge distillation
        new_global_model = fine_tune_on_remaining_data(new_global_model, train_loader, forgotten_labels=[5], device=device, num_epochs=5)


        # global_model = train_distilled(new_global_model, train_loader, device, temperature=2.0, num_epochs=5)
                
        print("\nEvaluating Global Model after unlearning 'shared_ab'...")
        acc_after, f1_after, report_after = evaluate_model(new_global_model, test_loader, device, None, unlearning_step=True)
        
        print("Running Unlearning Verification after Forgetting...")
        loss_after = membership_inference_attack(new_global_model, train_loader, unlearning_step=True)
        confi_forgotten_after, confi_unseen_after = confidence_score_difference(new_global_model, train_loader, test_loader, unlearning_step=True)
        mia_acc_after = adversarial_mia_attack(new_global_model, train_loader, test_loader, device, unlearning_step=True)
        
        plot_report_diff(report_before, report_after, "MNIST")
        plot_unlearning_findings(loss_before, loss_after, confi_forgotten_before, confi_forgotten_after, confi_unseen_before, confi_unseen_after, mia_acc_before, mia_acc_after)

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    track_resource_usage(0, "Complete Execution")
    print(f"Total execution time: {total_time:.2f} seconds")



# # To investigate further, you could:

# # Enable more aggressive unlearning: Try weighting the aggregation so that the forgotten model’s weights are entirely removed rather than averaged in.
# # Check the global model’s classifier parameters: Compare the aggregated weights before and after unlearning to see if there is a significant change.
# # Monitor label-specific metrics: Look specifically at precision, recall, and F1 for label 5 during unlearning to see if they change over time.
# # In summary, if label 5 performance isn’t dropping as expected, it could be due to the way the unlearning is implemented (averaging dilutes the effect) or because the global model is compensating during fine‑tuning. You might need a more aggressive or targeted unlearning strategy to see a significant drop.