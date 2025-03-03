# import tensorflow as tf

# # Define the top model for features 1 and 2
# def create_top_model():
#     input_top = tf.keras.Input(shape=(2,), name="top_features")  # Features 1 and 2
#     x = tf.keras.layers.Dense(16, activation="relu", name="top_dense1")(input_top)
#     x = tf.keras.layers.Dense(8, activation="relu", name="top_dense2")(x)
#     output_top = tf.keras.layers.Dense(4, activation="relu", name="top_output")(x)
#     return tf.keras.Model(inputs=input_top, outputs=output_top, name="TopModel")

# # Define the bottom model for feature 3
# def create_bottom_model_feature3():
#     input_bottom = tf.keras.Input(shape=(1,), name="bottom_feature3")  # Feature 3
#     x = tf.keras.layers.Dense(8, activation="relu", name="bottom_dense1")(input_bottom)
#     output_bottom = tf.keras.layers.Dense(4, activation="relu", name="bottom_output")(x)
#     return tf.keras.Model(inputs=input_bottom, outputs=output_bottom, name="BottomModelFeature3")

# # Define the bottom model for feature 4
# def create_bottom_model_feature4():
#     input_bottom = tf.keras.Input(shape=(1,), name="bottom_feature4")  # Feature 4
#     x = tf.keras.layers.Dense(8, activation="relu", name="bottom_dense1")(input_bottom)
#     output_bottom = tf.keras.layers.Dense(4, activation="relu", name="bottom_output")(x)
#     return tf.keras.Model(inputs=input_bottom, outputs=output_bottom, name="BottomModelFeature4")

# # Define the coordinator model for features 1, 2, and 3
# def create_coordinator_model_1(top_model, bottom_model):
#     input_top = top_model.input
#     input_bottom = bottom_model.input

#     concatenated = tf.keras.layers.Concatenate(name="concat1")([top_model.output, bottom_model.output])
#     x = tf.keras.layers.Dense(16, activation="relu", name="coordinator_dense1")(concatenated)
#     output = tf.keras.layers.Dense(1, activation="sigmoid", name="final_output1")(x)

#     return tf.keras.Model(inputs=[input_top, input_bottom], outputs=output, name="CoordinatorModel1")

# # Define the coordinator model for features 1, 2, and 4
# def create_coordinator_model_2(top_model, bottom_model):
#     input_top = top_model.input
#     input_bottom = bottom_model.input

#     concatenated = tf.keras.layers.Concatenate(name="concat2")([top_model.output, bottom_model.output])
#     x = tf.keras.layers.Dense(16, activation="relu", name="coordinator_dense2")(concatenated)
#     output = tf.keras.layers.Dense(1, activation="sigmoid", name="final_output2")(x)

#     return tf.keras.Model(inputs=[input_top, input_bottom], outputs=output, name="CoordinatorModel2")

# # Define the final overarching coordinator model
# def create_final_coordinator_model(coordinator1, coordinator2):
#     input_top_1 = coordinator1.input[0]  # Top input for Coordinator 1
#     input_bottom_1 = coordinator1.input[1]  # Bottom input for Coordinator 1 (Feature 3)
#     input_bottom_2 = coordinator2.input[1]  # Bottom input for Coordinator 2 (Feature 4)

#     # Outputs from the two coordinator models
#     output1 = coordinator1.output
#     output2 = coordinator2.output

#     # Concatenate the outputs
#     concatenated = tf.keras.layers.Concatenate(name="final_concat")([output1, output2])

#     # Final prediction layer
#     x = tf.keras.layers.Dense(16, activation="relu", name="final_dense1")(concatenated)
#     final_output = tf.keras.layers.Dense(1, activation="sigmoid", name="final_output")(x)

#     return tf.keras.Model(inputs=[input_top_1, input_bottom_1, input_bottom_2], outputs=final_output, name="FinalCoordinatorModel")

# # Create the models
# top_model = create_top_model()
# bottom_model_3 = create_bottom_model_feature3()
# bottom_model_4 = create_bottom_model_feature4()

# coordinator1 = create_coordinator_model_1(top_model, bottom_model_3)
# coordinator2 = create_coordinator_model_2(top_model, bottom_model_4)

# final_model = create_final_coordinator_model(coordinator1, coordinator2)

# # Compile the final model
# final_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Display the model summary
# final_model.summary()

# # Example usage with dummy data
# import numpy as np

# # Dummy data
# X_top = np.random.random((100, 2))  # 100 samples, 2 features (Top)
# X_bottom_3 = np.random.random((100, 1))  # 100 samples, 1 feature (Feature 3)
# X_bottom_4 = np.random.random((100, 1))  # 100 samples, 1 feature (Feature 4)
# y = np.random.randint(0, 2, size=(100, 1))  # Binary target

# # Train the model
# final_model.fit([X_top, X_bottom_3, X_bottom_4], y, epochs=10, batch_size=16)








import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import datasets, transforms

from torch.utils.data import DataLoader, TensorDataset

# from Prototype3_MNIST import devertiFL

class LocalModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LocalModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc_out(x)

class TopModel(nn.Module):
    def __init__(self, input_dim, top_hidden_dim):
        super(TopModel, self).__init__()
        self.fc = nn.Linear(input_dim, top_hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))

class BottomModel(nn.Module):
    def __init__(self, input_dim, bottom_hidden_dim):
        super(BottomModel, self).__init__()
        self.fc = nn.Linear(input_dim, bottom_hidden_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.fc(x))

# Shared model based in De-VertiFL's design.
class SharedModel(nn.Module):
    def __init__(self, top_model: nn.Module, bottom_model: nn.Module):
        super(SharedModel, self).__init__()
        self.top_model = top_model
        self.bottom_model = bottom_model
        all_shared_models = devertiFL()
        
    def forward(self, x):
        # P1 applies its top model on its local features
        top_out = self.top_model(x)
        # The collaborating party applies its bottom model (here we use x as a placeholder)
        bottom_out = self.bottom_model(x)
        # For the purpose of the global model, we assume the raw outputs are concatenated.
        # (This design decision means updates on one shared model's top part don't affect the other.)
        return torch.cat((top_out, bottom_out), dim=1)

# Global model that uses the raw outputs of the shared models as input.
# Its input dimension will be adjusted to match P1's original input features.
class GlobalModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(GlobalModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.hidden_layer = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(0.5)

        
    def forward(self, x):
        hidden_output = self.hidden_layer(self.input_layer(x))
        hidden_output = self.dropout(hidden_output)
        return hidden_output
    
    def predict(self, hidden_output):
        output = self.output_layer(hidden_output)
        return output

################################## deveriFL
def generate_hidden_outputs(model, data, device):
    hidden_outputs = []
 
    hidden_output = model(data.to(device))
    hidden_outputs.append(hidden_output)
    return hidden_outputs


def evaluate(model, device, test_loader):
    
    model.eval()
    expected_input_size = 392  # The input size expected by the model

    with torch.no_grad():
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            
            if data.shape[1] > expected_input_size:
                data = data[:, :expected_input_size]  # Trim excess features
            elif data.shape[1] < expected_input_size:
                padded_data = torch.zeros(data.shape[0], expected_input_size, device=device)
                padded_data[:, :data.shape[1]] = data  # Copy available features
                data = padded_data  # Use padded data

            hidden_outputs = generate_hidden_outputs(model, data, device)
            
            combined_hidden_outputs = torch.cat(hidden_outputs, dim=1)
            outputs = torch.zeros(data.size(0), 10, device=device)
           
            output = model.predict(combined_hidden_outputs)
            outputs += output

            test_loss += nn.CrossEntropyLoss(reduction='sum')(outputs, target).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return accuracy

################################## deveriFL

def import_hidden_outputs(file_path="hidden_outputs.csv"):
    df = pd.read_csv(file_path)
    targets = torch.tensor(df['target'].values, dtype=torch.long)
    hidden_outputs = torch.tensor(df.drop(columns=['target']).values, dtype=torch.float32)
    
    print(f"Hidden outputs and targets loaded from {file_path}")
    return hidden_outputs, targets


def vertical_partition_mnist(train_dataset, num_parties=4):
    assert num_parties == 4, "This function is designed for 4 participants."
    
    num_features = 784  # Total features (28x28)
    
    # Define feature split percentages
    sizes = [0.5, 0.2, 0.2, 0.1]  # 50%, 20%, 20%, 10%
    split_sizes = [int(num_features * s) for s in sizes]

    # Ensure sum is exactly 784 (handling rounding)
    split_sizes[-1] = num_features - sum(split_sizes[:-1])

    # Convert dataset to tensors
    all_images = torch.stack([img.view(-1) for img, _ in train_dataset])  # Shape: (60000, 784)
    all_labels = torch.tensor([label for _, label in train_dataset])  # Shape: (60000,)

    # Generate random feature indices
    indices = torch.randperm(num_features)  # Shuffle feature indices randomly
    feature_splits = []
    start = 0
    for size in split_sizes:
        feature_splits.append(indices[start:start+size])
        start += size

    # Create TensorDatasets for each participant
    partitioned_datasets = [TensorDataset(all_images[:, idxs], all_labels) for idxs in feature_splits]
    return partitioned_datasets, split_sizes


# MNIST data
def load_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def create_global_model():
    train_dataset, test_dataset = load_mnist_data()
    partitioned_datasets, splits = vertical_partition_mnist(train_dataset)
    print("***splits***")
    print(splits)   
    # all_shared_models, _ = devertiFL(3, partitioned_datasets, splits)
    inputs, targets = import_hidden_outputs("hidden_outputs_participants_test.csv")
    epochs=10
    lr=0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    projection_layer = torch.nn.Linear(60, 392).to(device)
    padded_input = projection_layer(inputs).detach()  # Detach from computation graph


    
    # global_model = GlobalModel(all_shared_models[0].layers[0].input_shape[0], 10)
    global_model = GlobalModel(392, 60, 10)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = global_model(padded_input)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    print("Training completed.")
    
    
    
    # Testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    accuracy = evaluate(global_model, device, test_loader)
    print(f"Accuracy: {accuracy:.2f}%")    





if __name__ == "__main__":
    create_global_model()