import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import os

# Shared model from which we will use the hidden outputs
class SharedModel(nn.Module):
    def __init__(self, input_size=392, hidden_size=256, num_classes=10):  
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
    def __init__(self, hidden_size=256, num_classes=10):
        super(GlobalModel, self).__init__()
        input_size = hidden_size * 3 #<- 3 because in this case we have 3 shared models, maybe change to a var 
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, h1, h2, h3):
        x = torch.cat((h1, h2, h3), dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def split_into_local_models(mnist_data):
    m1 = mnist_data[:, :196]  
    m2 = mnist_data[:, 196:392]  
    m3 = mnist_data[:, 392:588]  
    m4 = mnist_data[:, 588:]  
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
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            m1, m2, m3, m4 = split_into_local_models(images)

            input_ab = torch.cat((m1, m2), dim=1).to(device)
            input_ac = torch.cat((m1, m3), dim=1).to(device)
            input_ad = torch.cat((m1, m4), dim=1).to(device)

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

        print(f"Epoch [{epoch+1}/{num_epochs}] - Shared Models Loss: {loss_ab.item():.4f}, {loss_ac.item():.4f}, {loss_ad.item():.4f}")

    # export hidden outs
    torch.save(torch.cat(hidden_outputs_ab, dim=0), "hidden_outputs_ab.pt")
    torch.save(torch.cat(hidden_outputs_ac, dim=0), "hidden_outputs_ac.pt")
    torch.save(torch.cat(hidden_outputs_ad, dim=0), "hidden_outputs_ad.pt")
    torch.save(torch.cat(labels_list, dim=0), "labels.pt")

    print("Hidden outputs and labels saved!")

def load_and_split_data(test_ratio=0.2):
    h_ab = torch.load("hidden_outputs_ab.pt")
    h_ac = torch.load("hidden_outputs_ac.pt")
    h_ad = torch.load("hidden_outputs_ad.pt")
    labels = torch.load("labels.pt")

    hidden_inputs = torch.cat((h_ab, h_ac, h_ad), dim=1)

    total_samples = hidden_inputs.size(0)
    test_size = int(total_samples * test_ratio)
    train_size = total_samples - test_size

    train_data = hidden_inputs[:train_size]
    test_data = hidden_inputs[train_size:]
    train_labels = labels[:train_size]
    test_labels = labels[train_size:]

    return train_data, test_data, train_labels, test_labels

def train_global_model(global_model, train_data, train_labels, device, num_epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    global_model.to(device)
    train_data, train_labels = train_data.to(device), train_labels.to(device)

    for epoch in range(num_epochs):
        for i in range(0, len(train_data), 64):  
            inputs = train_data[i:i+64]
            labels = train_labels[i:i+64]

            outputs = global_model(inputs[:, :256], inputs[:, 256:512], inputs[:, 512:])  
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def evaluate_global_model(global_model, test_data, test_labels, device):
    global_model.eval()
    test_data, test_labels = test_data.to(device), test_labels.to(device)

    with torch.no_grad():
        outputs = global_model(test_data[:, :256], test_data[:, 256:512], test_data[:, 512:])
        _, predicted = torch.max(outputs, 1)
        total = test_labels.size(0)
        correct = (predicted == test_labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test data: {accuracy:.2f}%')
    return accuracy

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

if __name__ == "__main__":
    main()
