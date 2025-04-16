from logging import log
from re import L
import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import random
import numpy as np
import copy
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import matplotlib.pyplot as plt
import csv
import time
import os
from google.colab import files
import os
from google.colab import drive
import pandas as pd
from math import ceil
import torch.optim as optim
import torch.nn.functional as F
import os
print("📁 Current working directory:", os.getcwd())

log_file_path = "client_task_log.csv"

########################################
# SEED
########################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


########################################
#history of clients
########################################
client_history = {}
training_times = {}

########################################
# Machine Learning Model (Net)
########################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted to match the output size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Output: (batch, 6, 12, 12)
        x = self.pool(torch.relu(self.conv2(x)))  # Output: (batch, 16, 4, 4)
        x = torch.flatten(x, 1)                  # Output: (batch, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



########################################
# EdgeDevice Class
########################################
class EdgeDevice:
    def __init__(self, device_id, trainloader, valloader):
        self.device_id = device_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net()  # Each device has its own local model

    def get_client(self):
      return FlowerClient(self.model, self.trainloader, self.valloader, self.device_id, self.device_id)


########################################
# EdgeServer Class
########################################
class EdgeServer:
    def __init__(self, server_id, devices: List[EdgeDevice]):
        self.server_id = server_id
        self.devices = devices
        self.model = Net()  # Each edge server has its own local model

    def aggregate(self):
        """Aggregate models from all connected devices."""
        total_samples = 0
        weighted_params = None

        for device in self.devices:
            client = device.get_client()
            parameters = client.get_parameters()
            num_samples = len(device.trainloader.dataset)

            if weighted_params is None:
                weighted_params = [num_samples * np.array(param) for param in parameters]
            else:
                for i, param in enumerate(parameters):
                    weighted_params[i] += num_samples * np.array(param)

            total_samples += num_samples

        # Average the parameters
        aggregated_params = [param / total_samples for param in weighted_params]
        return aggregated_params

########################################
# Parameter Utility Functions
########################################
def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    state_keys = list(net.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in zip(state_keys, parameters)})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def test(net, testloader):
    net.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy

########################################
# Adjust L and H After Each Round
########################################
client_previous_bounds = {}

def adjust_task_assignment(round_number, clients, selected_clients, log_file_path, alpha, r1, r2, training_times):
    """
    Linear drop workload adjustment strategy:
    - Linear Drop affordable workload to 0 .
    - Otherwise, sample workload based on recent training history.
    """
    global client_previous_bounds  # Access the global dictionary
    args = {'GLOBAL_ROUNDS': 50,
            'alpha': 0.95}
    
    client = next(c for c in clients if c.cid == selected_clients[0])
    # ✅ Step 1: Retrieve previous round bounds if client was selected before
    if client.cid in client_previous_bounds:
          L_tk_before, H_tk_before = client_previous_bounds[client.cid]  # Reuse previous values
    else:
          L_tk_before, H_tk_before = client.lower_bound, client.upper_bound  # Use initial values
    
    with open(log_file_path, "a") as log_file:
      for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):


            training_time = training_times.get(client.cid, 0)
            STAGE_INCREASING = "INCREASING"
            STAGE_ARISE = "ARISE"
            STAGE_START = "START"
            STAGE_STRAGELLER = "STRAGELLER"
            STAGE_STRETCH = "STRETCH"
            STAGE_DROPOUT = "DROPOUT"
            stage = None




            # ✅ Normal Worklaod pattern
            for client in clients:
              
              
              mu_k = np.random.uniform(50, 60)
              sigma_k = np.random.uniform(mu_k / 4, mu_k / 2)
              client.affordable_workload = np.random.normal(mu_k, sigma_k)
              if not hasattr(client, "affordable_workload_init"):
                client.affordable_workload_init = client.affordable_workload
              print(f"Client {client.cid} Workload: {client.affordable_workload:.2f}")
              client.affordable_workload_logged = round_number  # Mark as updated for this round
              

              # ✅ Initialize workload range only once
              if not hasattr(client, 'lower_bound'):
                  client.lower_bound = 10
              if not hasattr(client, 'upper_bound'):
                  client.upper_bound = 20
              if not hasattr(client, 'threshold'):
                  client.threshold = client.upper_bound

              # ✅ Step 2: Maintain previous bounds if client has been selected before
              if client.cid in client_previous_bounds:
                  client.lower_bound, client.upper_bound = client_previous_bounds[client.cid]

              # ✅ Step 3: Update threshold
              client.threshold = alpha * client.threshold + (1 - alpha) * client.affordable_workload
              print(f"Client {client.cid} Threshold: {client.threshold:.2f}")

              # ✅ Step 4: Adjust workload range based on conditions
              if client.affordable_workload > H_tk_before:
                  if client.threshold <= L_tk_before:
                      print("ARISE")
                      client.lower_bound += r2
                      client.upper_bound += r2
                      stage = STAGE_ARISE
                  elif L_tk_before < client.threshold <= H_tk_before:
                      print("STRETCH")
                      client.lower_bound += r1
                      client.upper_bound += r2
                      stage = STAGE_STRETCH
                  else:
                      print("START")
                      client.lower_bound += r1
                      client.upper_bound += r1
                      stage = STAGE_START
                  client.affordable_workload = H_tk_before
                  

              elif L_tk_before < client.affordable_workload <= H_tk_before:
                  
                  if client.threshold >= L_tk_before:
                      print("STRAGELLER")
                      client.lower_bound = min(client.lower_bound + r2, 0.5 * H_tk_before)
                      client.upper_bound = max(client.lower_bound + r2, 0.5 * H_tk_before)
                      stage = STAGE_STRAGELLER

                  elif L_tk_before < client.threshold <= H_tk_before:
                      print("STRAGELLER")
                      client.lower_bound = min(client.lower_bound + r1, 0.5 * H_tk_before)
                      client.upper_bound = max(client.lower_bound + r1, 0.5 * H_tk_before)
                      stage = STAGE_STRAGELLER
                
                  client.affordable_workload = L_tk_before
                  

              else:
                      client.lower_bound = 0.5 * L_tk_before
                      client.upper_bound = 0.5 * H_tk_before
                      client.affordable_workload = 0
                      stage = STAGE_DROPOUT

              # ✅ Step 5: Capture updated values after adjustment
              L_tk_after = client.lower_bound
              H_tk_after = client.upper_bound
              print(f"Client {client.cid} Updated Bounds: L={L_tk_after:.2f}, H={H_tk_after:.2f}")

              

              # ✅ Step 6: Store updated bounds for future reference
              client_previous_bounds[client.cid] = (L_tk_after, H_tk_after)
              print(f"Client {client.cid} Previous Bounds: L={L_tk_before:.2f}, H={H_tk_before:.2f}")


              # ✅ Step 8: Explicitly update bounds again
              client.lower_bound = L_tk_after
              client.upper_bound = H_tk_after
              print(f"Client {client.cid} Updated Bounds: L={L_tk_after:.2f}, H={H_tk_after:.2f}")
              

              
              
            return L_tk_before, H_tk_before, L_tk_after, H_tk_after, stage



########################################
# Trainig round adjustment
########################################
def compute_training_rounds(client_id, clients, base_k1):
    """
    Compute training rounds dynamically to ensure average k1 remains base_k1.
    Increase training rounds if needed.
    """
    client = next(c for c in clients if c.cid == client_id)
    training_rounds = max(10, int(round(base_k1 * (client.affordable_workload / 60), 2)))


    # Scale up if training time is too short
    return max(training_rounds, 20)  # Ensures at least 20 epochs



########################################
# Energy Consumption
########################################
def compute_energy(
    affordable_workload, train_time_sample, computation_power, transmitter_power, model_size_bits, channel_capacity):

    # Training time per sample
    total_training_time = affordable_workload * train_time_sample

    # Step 1: Computation energy
    energy_comp = computation_power * total_training_time

    # Step 2: Communication energy (model upload once per round)
    transmission_time = model_size_bits / channel_capacity
    energy_comm = transmitter_power * transmission_time

    # Step 3: Total energy
    total_energy = energy_comp + energy_comm

    return energy_comp, energy_comm, total_energy, total_training_time


########################################
# FlowerClient Class
########################################
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, testloader, valloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.cid = cid

        # ✅ Initialize affordable workload
        self.affordable_workload = self.initialize_affordable_workload()
        self.lower_bound = 10
        self.upper_bound = 20
        self.threshold = 20


    def initialize_affordable_workload(self):
        """Generate a client's affordable workload using a normal distribution."""
        mu_k = np.random.uniform(50, 60)  # Mean workload
        sigma_k = np.random.uniform(mu_k / 4, mu_k / 2)  # Standard deviation
        return max(0, np.random.normal(mu_k, sigma_k))  # Ensure workload is non-negative

    def reset_affordable_workload(self):
        """Reset affordable workload when a client recovers from failure."""
        self.affordable_workload = 0
        print(f"🔄 Client {self.cid} recovered. Affordable workload reset to 0.")

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in zip(state_keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        global client_history
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        client_id = self.cid

        if client_id not in client_history:
            client_history[client_id] = {"training_time": []}

        num_epochs = config.get("num_epochs", 60)  # Default to 60 if missing
        num_epochs = int(num_epochs) if num_epochs else 60  # Ensure integer



        start_time = time.time()
        for epoch in range(num_epochs):
            for inputs, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        end_time = time.time()
        training_time = end_time - start_time

        training_times[client_id] = training_time

        client_history[client_id]["training_time"].append(training_time)
        print(f"✅ Client {client_id}: Training completed in {training_time:.2f} sec.")

        return self.get_parameters(), len(self.trainloader.dataset), {"training_time": training_time}


########################################
# Hierarchical Federated Learning with Flower
#######################################
def HierFL(args, trainloaders, valloaders, testloader):
    global client_history
    global_model = Net()
    global_weights = get_parameters(global_model)
    computation_power = 0.5        # in Watts
    transmitter_power = 0.1        # in Watts
    train_time_sample = 0.01       # seconds per sample
    model_size_bits = 100_000      # bits
    channel_capacity = 1_000_000   # bits/second
    samples_per_epoch = 1000
    r1 = args['r1']
    r2 = args['r2']


    log_file_path = "client_task_log.csv"

    # ✅ Initialize Failure Tracking (Only once)
    training_times = {}  # Store training times for each client

    # ✅ Initialize Clients Once
    clients = [
        FlowerClient(
            model=Net(),
            trainloader=trainloaders[i],
            testloader=testloader,
            valloader=valloaders[i],
            cid=i
        )
        for i in range(args['NUM_DEVICES'])
    ]

    # ✅ Initialize Edge Devices
    edge_devices = [EdgeDevice(i, trainloaders[i], valloaders[i]) for i in range(args['NUM_DEVICES'])]
    num_edge_servers = args['NUM_EDGE_SERVERS']
    edge_servers = []
    devices_per_server = len(edge_devices) // num_edge_servers

    for i in range(num_edge_servers):
        start_idx = i * devices_per_server
        devices = edge_devices[start_idx:] if i == num_edge_servers - 1 else edge_devices[start_idx: start_idx + devices_per_server]
        edge_servers.append(EdgeServer(i, devices))



    # ✅ Define Evaluation Function
    def evaluate_fn(server_round, parameters, config):
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    # ✅ Define Federated Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
    )

    # ✅ Start Federated Learning Simulation
    #fl.simulation.start_simulation(
        #client_fn=lambda cid: FlowerClient(
            #model=Net(),
            #trainloader=trainloaders[int(cid)],
            #valloader=valloaders[int(cid)],
            #testloader=testloader,
            #cid=int(cid)
        #),
        #num_clients=len(trainloaders),
        #config=fl.server.ServerConfig(num_rounds=args['GLOBAL_ROUNDS']),
        #strategy=strategy
    #)

    # ✅ Ensure CSV file exists before starting logging
    if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("Round,Client,Status,TrainingTime,LowerBound_before,UpperBound_before,LowerBound_after,UpperBound_after,AffordableWorkloadInitialize,ClientAffordableWorkload,NumEpoch,Theta,State\n")


    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):
        start_time = time.time()

        # ✅ Select only available clients for training
        available_clients = [cid for cid in range(args["NUM_DEVICES"])]
        if not available_clients:
            print(f"⚠️ No available clients in round {round_number}. Skipping round.")
            continue

        selected_clients = available_clients

        L_tk_before, H_tk_before, L_tk_after, H_tk_after, stage = adjust_task_assignment(
            round_number=round_number,
            clients=clients,
            selected_clients=selected_clients,
            alpha=args['alpha'],
            r1=args['r1'],
            r2=args['r2'],
            training_times=training_times,  # ideally tracked over rounds
            log_file_path=log_file_path,
            

        )


        for client_id in selected_clients:

            num_epochs = compute_training_rounds(client_id, clients, args['base_k1'])
            client = clients[client_id]  # ✅ Correctly reference the client

            # ✅ Train the client before logging
            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})
            print(f"✅ Client {client_id} trained for {num_epochs} epochs.")
            training_time = train_metrics["training_time"]
            training_times[client_id] = training_time
            total_training_time = sum(training_times.values())

            # ✅ Compute energy consumption
            affordable_workload = client.affordable_workload
            energy_comp, energy_comm, total_energy, training_time = compute_energy(
              affordable_workload,
              train_time_sample,
              computation_power,
              transmitter_power,
              model_size_bits,
              channel_capacity
    )



            # ✅ Log training clients with workload details
            AffordableWorkloadInitialize = client.affordable_workload_init
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{round_number},{client_id},TRAINING,{training_time},{L_tk_before},{H_tk_before},{L_tk_after},{H_tk_after},{AffordableWorkloadInitialize:.2f},{client.affordable_workload:.2f},{num_epochs},{client.threshold:.2f},{stage},{energy_comp:.2f}, {energy_comm:.2f}, {total_energy:.2f}, {total_training_time:.2f}\n")


        # ✅ Edge Aggregation
        if round_number % args["k2"] == 0:
            print(f"🔹 Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
                aggregated_params = edge_server.aggregate()
                set_parameters(edge_server.model, aggregated_params)

        # ✅ Global Aggregation
        if round_number % (args["k1"] * args["k2"]) == 0:
            print(f"🌍 Aggregating at GLOBAL SERVER (every {args['k1'] * args['k2']} rounds)")
            global_weights = get_parameters(global_model)
            set_parameters(global_model, global_weights)

    print(f"✅ Finished {args['GLOBAL_ROUNDS']} rounds. Check {log_file_path} for logs.")


########################################
# Main
#######################################

def main():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    def load_datasets(num_clients: int, batch_size: int = 32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3 channels
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(SEED))
        split = torch.split(indices, len(train_dataset) // num_clients)

        trainloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED)) for s in split]
        valloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=False) for s in split]
        testloader = DataLoader(test_dataset, batch_size=batch_size)
        return trainloaders, valloaders, testloader

    args = {
        'NUM_DEVICES': 20,
        'NUM_EDGE_SERVERS': 5,
        'GLOBAL_ROUNDS':10,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'CLIENT_FRACTION': 0.2,
        'EVALUATE_FRACTION': 0.2,
        'alpha': 0.95,
        'r1': 3,
        'r2': 1,
        'base_k1': 60,
        'num_epochs': 10,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()



