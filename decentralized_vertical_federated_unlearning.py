import glob
from tracemalloc import start
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import psutil
import time
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
 

# Shared model from which we will use the hidden outputs
class SharedModel(nn.Module):
    def __init__(self, input_size=392, hidden_size=261, num_classes=10):  
        super(SharedModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        hidden_output = self.relu(self.fc1(x)) 
        classification_output = self.fc_out(hidden_output)
        return hidden_output, classification_output

# Global Model which will has the input size of x times hidden outputs 
class GlobalModel(nn.Module):
    def __init__(self, hidden_size=261, num_classes=10):
        super(GlobalModel, self).__init__()
        input_size = hidden_size * 3 #<- 3 because in this case we have 3 shared models, maybe change to a var 
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, h1, h2, h3):
        x = torch.cat((h1, h2, h3), dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
    
class ParticipantModel(nn.Module):
    def __init__(self, hidden_size=261, num_classes=10):
        super(ParticipantModel, self).__init__()
        input_size = hidden_size * 3
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.ReLU()
        self.output_layer = nn.Linear(input_size, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, h1, h2, h3):
        x = torch.cat((h1, h2, h3), dim=1)
        hidden_output = self.hidden_layer(self.input_layer(x))
        hidden_output = self.dropout(hidden_output)
        return hidden_output

    def predict(self, combined_hidden_outputs):
        output = self.output_layer(combined_hidden_outputs)
        return output
    
def track_resource_usage(epoch, phase="Training"):
    """Prints resource usage stats"""
    cpu_mem = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
    gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0  # Convert to GB
    print(f"[{phase} - Epoch {epoch+1}] CPU Memory Used: {cpu_mem:.2f} GB, GPU Memory Used: {gpu_mem:.2f} GB")


def split_into_local_models(mnist_data, labels, nonIID=False):
    if nonIID:
        return non_IID_partition_data(mnist_data, labels)
    m1 = mnist_data[:, :196]  
    m2 = mnist_data[:, 196:392]  
    m3 = mnist_data[:, 392:588]  
    m4 = mnist_data[:, 588:]  
    print(f"m1: {m1.shape}, m2: {m2.shape}, m3: {m3.shape}, m4: {m4.shape}")

    return m1, m2, m3, m4

from collections import Counter

def non_IID_partition_data(data, labels, batch_size=64):
    seven_indices = (labels == 7).nonzero(as_tuple=True)[0]
    non_seven_indices = (labels != 7).nonzero(as_tuple=True)[0]

    # Extract data partitions
    m2 = data[seven_indices, 196:392]  # Only '7' samples
    m1 = data[non_seven_indices, :196]
    m3 = data[non_seven_indices, 392:588]
    m4 = data[non_seven_indices, 588:]

    # Extract corresponding labels for each partition
    labels_m1 = labels[non_seven_indices]
    labels_m2 = labels[seven_indices]
    labels_m3 = labels[non_seven_indices]
    labels_m4 = labels[non_seven_indices]

    # Print label distributions
    print("\n🔹 Label Distributions in Each Partition:")
    print(f"m1: {dict(Counter(labels_m1.tolist()))}")
    print(f"m2 (Only 7s): {dict(Counter(labels_m2.tolist()))}")  # Should contain only {7: count}
    print(f"m3: {dict(Counter(labels_m3.tolist()))}")
    print(f"m4: {dict(Counter(labels_m4.tolist()))}")

    # Ensure batch size alignment
    min_samples = min(m1.shape[0], m3.shape[0], m4.shape[0])

    def resample_to_match(tensor, target_size):
        """Resample tensor to match the target size."""
        indices = torch.randint(0, tensor.shape[0], (target_size,))
        return tensor[indices]

    new_size = (min_samples // batch_size) * batch_size  # Make divisible by batch size
    m1, m3, m4 = m1[:new_size], m3[:new_size], m4[:new_size]
    m2 = resample_to_match(m2, new_size)

    print(f"Final Sizes -> m1: {m1.shape}, m2: {m2.shape}, m3: {m3.shape}, m4: {m4.shape}")
    return m1, m2, m3, m4


def train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5):
    shared_ab.to(device)
    shared_ac.to(device)
    shared_ad.to(device)

    optimizer_ab = optim.Adam(shared_ab.parameters(), lr=0.001)
    optimizer_ac = optim.Adam(shared_ac.parameters(), lr=0.001)
    optimizer_ad = optim.Adam(shared_ad.parameters(), lr=0.001)
    
    criterion = nn.CrossEntropyLoss()

    hidden_outputs_ab = []
    hidden_outputs_ac = []
    hidden_outputs_ad = []
    labels_list = []

    for epoch in range(num_epochs):
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            IID=True
            m1, m2, m3, m4 = split_into_local_models(images, labels, True)

            pad_size = m1.shape[0] - m2.shape[0]
            m2 = torch.cat((m2, torch.zeros((pad_size, m2.shape[1]), device=m2.device)), dim=0)  # Pad with zeros
            input_ab = torch.cat((m1, m2), dim=1).to(device)
            
            input_ac = torch.cat((m1, m3), dim=1).to(device)
            input_ad = torch.cat((m1, m4), dim=1).to(device)
            
            print(input_ab.shape, input_ac.shape, input_ad.shape)

            h_ab, pred_ab = shared_ab(input_ab)  
            h_ac, pred_ac = shared_ac(input_ac)  
            h_ad, pred_ad = shared_ad(input_ad)  

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

            # store hidden outputs for the global model
            hidden_outputs_ab.append(h_ab.cpu())
            hidden_outputs_ac.append(h_ac.cpu())
            hidden_outputs_ad.append(h_ad.cpu())
            labels_list.append(labels.cpu())

        epoch_time = time.time() - start_time
        track_resource_usage(epoch, "Shared Model Training")
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Losses: {loss_ab.item():.4f}, {loss_ac.item():.4f}, {loss_ad.item():.4f}")


    # export hidden outs
    torch.save(torch.cat(hidden_outputs_ab, dim=0), "./resources/hidden_outputs/hidden_outputs_ab.pt")
    torch.save(torch.cat(hidden_outputs_ac, dim=0), "./resources/hidden_outputs/hidden_outputs_ac.pt")
    torch.save(torch.cat(hidden_outputs_ad, dim=0), "./resources/hidden_outputs/hidden_outputs_ad.pt")
    torch.save(torch.cat(labels_list, dim=0), "labels.pt")

    print("Hidden outputs and labels saved!")

def load_and_split_data(test_ratio=0.2):
    """
    Dynamically loads existing hidden outputs and adjusts the dataset accordingly.
    Handles missing features when a client is forgotten.
    """
    # Load only available hidden representations
    hidden_outputs = []
    if os.path.exists("./resources/hidden_outputs/hidden_outputs_ab.pt"):
        hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ab.pt"))
    if os.path.exists("./resources/hidden_outputs/hidden_outputs_ac.pt"):
        hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ac.pt"))
    if os.path.exists("./resources/hidden_outputs/hidden_outputs_ad.pt"):
        hidden_outputs.append(torch.load("./resources/hidden_outputs/hidden_outputs_ad.pt"))

    if not hidden_outputs:
        raise ValueError("No valid hidden representations found. Did you forget all clients?")

    # Concatenate only the existing hidden outputs
    for output in hidden_outputs:
        output = output[:1000]
    hidden_inputs = torch.cat(hidden_outputs, dim=1)
    hidden_inputs = hidden_inputs[:1000]
    print("hidden inputs shape", hidden_inputs.shape)

    # Load labels
    labels = torch.load("labels.pt") if os.path.exists("labels.pt") else None
    if labels is None:
        raise ValueError("Labels file is missing.")

    labels = labels[:1000]
    
    # Determine split index
    total_samples = hidden_inputs.size(0)
    test_size = int(total_samples * test_ratio)
    train_size = total_samples - test_size

    # Split data
    train_data = hidden_inputs[:train_size]
    test_data = hidden_inputs[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    return train_data, test_data, train_labels, test_labels


def train_global_model(global_model, train_data, train_labels, device, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) #try batch 512 later
    
    global_model.to(device)
    global_model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} started at {start_time}")
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            num_sources = inputs.shape[1] // 261
            input_slices = [inputs[:, i*261:(i+1)*261] for i in range(num_sources)]
            
            # outputs = global_model(inputs[:, :261], inputs[:, 261:512], inputs[:, 512:])
            outputs = global_model(*input_slices)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_time = time.time() - start_time
        track_resource_usage(epoch, "Global Model Training")   
        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Loss: {loss.item():.4f}")



def evaluate_global_model(global_model, test_data, test_labels, device):
    global_model.eval()
    test_data, test_labels = test_data.to(device), test_labels.to(device)

    all_preds, all_targets = [], []

    with torch.no_grad():
        outputs = global_model(test_data[:, :256], test_data[:, 256:512], test_data[:, 512:])
        _, predicted = torch.max(outputs, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(test_labels.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average="macro")  # Macro-F1 for multi-class
    
    accuracy = 100 * (predicted == test_labels).sum().item() / test_labels.size(0)
    
    print(f'Accuracy on test data: {accuracy:.2f}%')
    print(f'F1 Score: {f1:.4f}')  # Print F1 Score

    return accuracy, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    shared_ab = SharedModel() # either use devertiFL or just straight up init the shared model
    shared_ac = SharedModel()
    shared_ad = SharedModel()

    if not os.path.exists("hidden_outputs_ab.pt"):
        print("Training Shared Models and Storing Hidden Outputs...")
        train_shared_models(shared_ab, shared_ac, shared_ad, train_loader, device, num_epochs=5)

    train_data, test_data, train_labels, test_labels = load_and_split_data()

    global_model = GlobalModel()
    print("Training Global Model...")
    train_global_model(global_model, train_data, train_labels, device, num_epochs=5)

    print("Evaluating Global Model...")
    evaluate_global_model(global_model, test_data, test_labels, device)
    

def forget_client(client_name):
    print(f"Forgetting {client_name} and its dataset contribution...")

    # Determine which files to remove
    if client_name == "shared_ab":
        remove_file = "./resources/hidden_outputs/hidden_outputs_ab.pt"
    elif client_name == "shared_ac":
        remove_file = "./resources/hidden_outputs/hidden_outputs_ac.pt"
    elif client_name == "shared_ad":
        remove_file = "./resources/hidden_outputs/hidden_outputs_ad.pt"
    else:
        print("Invalid client name! Choose from: 'shared_ab', 'shared_ac', 'shared_ad'")
        return

    # Delete the specified shared model's hidden representations
    if os.path.exists(remove_file):
        os.remove(remove_file)
        print(f"Removed {remove_file}")
    else:
        print(f"{remove_file} does not exist.")

    # Reload and split the dataset with the remaining features
    train_data, test_data, train_labels, test_labels = load_and_split_data()

    # Adjust the global model input size (reduce features)
    feature_count = train_data.shape[1] // 261
    print(f"Updating Global Model to accept {feature_count} feature sources instead of 3.")

    class UpdatedGlobalModel(nn.Module):
        def __init__(self, hidden_size=261, num_classes=10, num_sources=feature_count):
            super(UpdatedGlobalModel, self).__init__()
            input_size = hidden_size * num_sources
            self.fc1 = nn.Linear(input_size, 512)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(512, num_classes)

        def forward(self, *hidden_inputs):
            x = torch.cat(hidden_inputs, dim=1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    # Train the new global model with fewer inputs
    global_model = UpdatedGlobalModel(num_sources=feature_count)
    print("\n🚀 Retraining Global Model with fewer features...")
    train_global_model(global_model, train_data, train_labels, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    evaluate_global_model(global_model, test_data, test_labels, torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Forgetting process complete. Global model updated!")


# TODO: compare time perspective of retraining the whole federation vs only the global model
# TODO: log usage of resources
if __name__ == "__main__":
    start_time = time.time()
    main()
    forget_client("shared_ab")
    complete_time_with_removing = time.time() - start_time
    track_resource_usage(1, "Complete DevertiFL")
    print(f"Complete DevertiFL Execution Time with only retraining global model: {complete_time_with_removing:.2f}s")



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# import torch.cuda.amp as amp
# import time

# # Enable cuDNN optimization SPEED
# torch.backends.cudnn.benchmark = True

# def train_global_model(global_model, train_data, train_labels, device, num_epochs=5, batch_size=64, accumulate_steps=2):
#     """
#     Trains the global model with aggressive performance optimizations:
#     - Uses DataLoader with num_workers=8 for parallel loading
#     - Uses non-blocking GPU transfers
#     - Implements Gradient Accumulation for faster convergence
#     - Uses Mixed Precision Training (AMP) for speed-up
#     - Clears GPU cache periodically to avoid fragmentation
#     """

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(global_model.parameters(), lr=0.001)
    
#     # Mixed precision training (Automatic Mixed Precision)
#     scaler = amp.GradScaler()

#     # Optimized DataLoader
#     train_dataset = TensorDataset(train_data, train_labels)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

#     global_model.to(device)
#     global_model.train()

#     for epoch in range(num_epochs):
#         start_time = time.time()
#         total_loss = 0
#         optimizer.zero_grad()

#         for batch_idx, (inputs, labels) in enumerate(train_loader):
#             inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

#             # Dynamically determine number of input feature slices
#             num_sources = inputs.shape[1] // 261
#             input_slices = [inputs[:, i*261:(i+1)*261] for i in range(num_sources)]

#             with amp.autocast():  # Enable Mixed Precision for speed-up
#                 outputs = global_model(*input_slices)
#                 loss = criterion(outputs, labels)

#             # Accumulate gradients
#             scaler.scale(loss).backward()
#             total_loss += loss.item()

#             # Step optimizer every `accumulate_steps` batches
#             if (batch_idx + 1) % accumulate_steps == 0 or batch_idx == len(train_loader) - 1:
#                 scaler.step(optimizer)
#                 scaler.update()
#                 optimizer.zero_grad()

#         # Free up GPU cache every few epochs to prevent slowdown
#         if epoch % 3 == 0:
#             torch.cuda.empty_cache()

#         epoch_time = time.time() - start_time
#         track_resource_usage(epoch, "Global Model Training")   
#         print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s, Avg Loss: {total_loss / len(train_loader):.4f}")
