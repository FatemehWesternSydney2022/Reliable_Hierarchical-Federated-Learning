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
from math import floor
print("📁 Current working directory:", os.getcwd())

log_file_path = "client_task_log.csv"
log_file = "client_failure_prediction.csv"

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
training_epochs = {}

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
# LSTM prediction
########################################
class LSTMFailurePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(LSTMFailurePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last output
        return out

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
    def __init__(self, server_id, devices: List['EdgeDevice'], backup_server: 'EdgeServer' = None):
        self.server_id = server_id
        self.devices = devices  # List of EdgeDevice instances
        self.model = Net()      # Each edge server has its own local model
        self.failed = False     # Track failure status
        self.backup_server = backup_server  # One backup edge server

    def is_failed(self):
        return self.failed

    def handle_failure(self, edge_servers: List['EdgeServer'], client_to_server_map: dict):
        """Mark self as failed and reassign all clients to the backup server."""
        self.failed = True
        print(f"⚠️ Edge server {self.server_id} has failed. Reassigning clients...")

        for backup_server in self.backup_servers:
          if backup_server != self and not backup_server.is_failed():
              for device in self.devices:
                  if hasattr(device, "client") and device.client is not None:
                    backup_server.add_device(device)
                    client_to_server_map[device.client.cid] = backup_server.server_id
                    print(f"✅ Client {device.client.cid} reassigned to EdgeServer {backup_server.server_id}")
                  else:
                    print(f"⚠️ Device {device.device_id} has no associated client.")

              self.devices.clear()
              print(f"✅ Clients reassigned to backup server {backup_server.server_id}.")
              return  # Exit after successful reassignment

        print(f"❌ Backup server not available for server {self.server_id}.")

    def add_device(self, device: 'EdgeDevice'):
        """Add a device (client wrapper) to this server."""
        self.devices.append(device)

    def remove_device(self, device: 'EdgeDevice'):
        """Remove a device from this server."""
        self.devices.remove(device)

    def aggregate(self):
      """Aggregate models from all connected devices."""
      total_samples = 0
      weighted_params = None

      for device in self.devices:
          client = device.get_client()
          if client is None:
              print(f"⚠️ Device {device.device_id} has no client. Skipping.")
              continue

          parameters = client.get_parameters()
          if parameters is None:
              print(f"⚠️ Client {client.cid} has no parameters. Skipping.")
              continue

          num_samples = len(device.trainloader.dataset)
          if weighted_params is None:
              weighted_params = [num_samples * np.array(param) for param in parameters]
          else:
              for i, param in enumerate(parameters):
                  weighted_params[i] += num_samples * np.array(param)

          total_samples += num_samples

      if total_samples == 0 or weighted_params is None:
          print(f"⚠️ Edge server {self.server_id} has no valid data for aggregation.")
          return None

      # Average the parameters
      aggregated_params = [param / total_samples for param in weighted_params]
      return aggregated_params



########################################
# Random Edge Server Backup
########################################
def assign_random_backups(edge_servers: List[EdgeServer], edge_devices: List[EdgeDevice]):
    for server in edge_servers:
        possible_backups = [s for s in edge_servers if s.server_id != server.server_id]
        server.backup_server = random.choice(possible_backups)
        print(f"Server {server.server_id} assigned backup server {server.backup_server.server_id}.")

########################################
# Initilize edge server with their backup edge server
########################################
def initialize_edge_servers_with_backups(edge_devices: List[EdgeDevice], num_servers: int) -> Tuple[List[EdgeServer], dict]:
    """
    Initializes edge servers with backups and assigns clients.
    Also builds the initial client-to-server mapping.
    """
    edge_servers = [EdgeServer(server_id=i, devices=[]) for i in range(num_servers)]
    client_to_server_map = {}

    # ✅ Assign one random backup for each edge server (excluding itself)
    for server in edge_servers:
        possible_backups = [s for s in edge_servers if s.server_id != server.server_id]
        server.backup_servers = [random.choice(possible_backups)]

    # ✅ Assign devices evenly across edge servers
    devices_per_server = len(edge_devices) // num_servers
    for i, server in enumerate(edge_servers):
        start = i * devices_per_server
        end = None if i == num_servers - 1 else (i + 1) * devices_per_server
        server.devices = edge_devices[start:end]

        # ✅ Update client-to-server mapping
        for device in server.devices:
            if hasattr(device, "client") and device.client is not None:
                client_to_server_map[device.client.cid] = server.server_id

    return edge_servers, client_to_server_map


########################################
# Edge Server's Failure
########################################
def simulate_edge_failures(edge_servers: List[EdgeServer], failed_servers: set, client_to_server_map,failure_rate: float = 0.05):
    available_servers = [s for s in edge_servers if not s.is_failed()]

    if not available_servers:
        print("⚠️ All edge servers have failed. Skipping failure simulation.")
        return

    num_to_fail = max(1, int(len(edge_servers) * failure_rate))  # At least one server fails
    servers_to_fail = random.sample(available_servers, min(num_to_fail, len(available_servers)))

    for server_to_fail in servers_to_fail:
        server_to_fail.handle_failure(edge_servers,client_to_server_map)
        failed_servers.add(server_to_fail.server_id)



########################################
# hold each client’s ID
########################################

class Client:
    def __init__(self, cid):
        self.cid = cid  # Client ID

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
# Avergae Epoch time calculation
########################################
def calculate_avg_epochs_per_client(training_epochs, num_clients):
    total_epochs = sum(training_epochs.values())
    avg_epochs_per_client = total_epochs / num_clients if num_clients > 0 else 0
    return avg_epochs_per_client



########################################
# Adjust L and H After Each Round
########################################
client_previous_bounds = {}


def adjust_task_assignment(round_number, clients, selected_clients, log_file_path, alpha, r1, r2, training_times, failure_history, predicted_failure_time, lstm_model, num_epochs,args, avg_epoch):

    global client_previous_bounds  # Access the global dictionary
    args = {'GLOBAL_ROUNDS': 50,
            'alpha': 0.95,
            'base_r1' : 0.2,
            'base_r2': 0.1,
            }


    client = next(c for c in clients if c.cid == selected_clients[0])

    failure_history = client.failure_history


    # Adjust r1 and r2 based on average epoch time
    r1 = args['base_r1'] * avg_epoch  # r1 as a percentage of average epoch time
    print(f"Adjusted r1: {r1}")
    r2 = args['base_r2'] * avg_epoch  # r2 as a percentage of average epoch time
    print(f"Adjusted r2: {r2}")


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
              AffordableWorkloadInitilize = client.affordable_workload
              print(f"Client {client.cid} Workload: {client.affordable_workload:.2f}")
              NumEpoch = floor(AffordableWorkloadInitilize)
              client.num_epochs = NumEpoch  # Store in the client object
              training_epochs[client.cid] = NumEpoch  # Also store in global or shared dictionary


              client.affordable_workload_logged = round_number  # Mark as updated for this round


              # ✅ Initialize workload range only once
              if not hasattr(client, 'lower_bound'):
                  client.lower_bound = 10
              if not hasattr(client, 'upper_bound'):
                  client.upper_bound = 20
              if not hasattr(client, 'threshold'):
                  client.threshold = client.threshold = client.upper_bound

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

              with open(log_file_path, "a") as log_file:
                log_file.write(f"{round_number},{client.cid},{L_tk_before},{H_tk_before},{L_tk_after},{H_tk_after},{AffordableWorkloadInitilize},{client.affordable_workload:.2f},{NumEpoch:.2f},{client.threshold:.2f},{r1:.2f},{r2:.2f},{stage}\n")




            return L_tk_before, H_tk_before, L_tk_after, H_tk_after, stage



########################################
# Trainig round adjustment
########################################
def compute_training_rounds(client_id, clients, base_k1):
    client = next(c for c in clients if c.cid == client_id)
    training_rounds = int(round(base_k1 * (client.affordable_workload / 60), 2))

    # ✅ Use initial (pre-adjusted) workload as a lower bound
    min_epochs = int(client.initial_affordable_workload)
    return  max(min_epochs, training_rounds)


########################################
# Random Failure Simulation
########################################
client_failure_history = {}  # {client_id: [failure_rounds]}

def simulate_failures(args, unavailability_tracker, failure_log, round_number, training_times, selected_clients, lstm_model, failure_history):
    """Simulates client failures and updates the failure log dynamically per round."""
     #Identify clients who are available (not currently failing)

    recovered_this_round = set()

    available_clients = [
        cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0
    ]

    num_failures = int(len(available_clients) * args['FAILURE_RATE'])
    num_failures = max(1, num_failures)  # Ensure at least one client fails



    # Ensure failure is only assigned to NEW available clients, not already failing ones
    failing_clients = [
        cid for cid in available_clients if cid not in [c[0] for c in failure_log]
    ]

    failing_clients = random.sample(failing_clients, min(num_failures, len(failing_clients)))

    # Assign failure durations and update log
    new_failures = []
    avg_training_time = 0
    recovery_time_remaining = 0

    for client_id in failing_clients:
        failure_duration = random.randint(1, args['FAILURE_DURATION'])
        training_times = {client_id: training_times.get(client_id, 0) for client_id in selected_clients}
        avg_training_time = np.mean(list(training_times.values())) if training_times else 0  # Assign failure duration
        recovery_time_remaining = failure_duration - avg_training_time # Initially set full failure time


        unavailability_tracker[client_id] = 1  # Mark as unavailable
        new_failures.append([client_id, failure_duration, recovery_time_remaining])

        # Log the current failure timestamp for the client
        current_timestamp = time.time()  # You can use the current round number as a timestamp or real-time
        if client_id not in failure_history:
            failure_history[client_id] = []
        failure_history[client_id].append(current_timestamp)  # Add the failure timestamp
        print(failure_history)

        with open(log_file_path, "a") as log_file:
            log_file.write(f"{round_number},{client_id},FAILED,{failure_duration},{recovery_time_remaining},{avg_training_time},0,0,0,0,0,{failure_duration}:\n")
            log_file.flush()

    # Append new failures to log
    failure_log.extend(new_failures)


    # ✅ **Update recovery time for all currently failing clients**
    for client in failure_log:
        client[2] -= avg_training_time # Reduce recovery time by failure duration
        client[2] = max(0, client[2])  # Ensure it doesn’t go negative
        if client[2] == 0:
            unavailability_tracker[client[0]] = 0  # Mark as available
            recovered_this_round.add(client[0])  # Add to set of recovered clients

    # ✅ **Recover clients whose recovery time reaches 0**
    recovered_clients = [c for c in failure_log if c[2] == 0]
    for client in recovered_clients:
        client_id = client[0]
        unavailability_tracker[client_id] = 0  # Mark client as available
        failure_log.remove(client)  # Remove from failure log

        with open(log_file_path, "a") as log_file:
            log_file.write(f"{round_number},{client_id},RECOVERED,0,0,0,0,0,0,0,0:\n")
            log_file.flush()

    return failure_log, recovered_this_round

########################################
# Total Workload Calculation
########################################
def calculate_total_workload(selected_clients):
    """
    Calculate the total workload for the selected clients in a round.

    selected_clients: list of selected client objects

    Returns: total workload for the round (sum of all affordable workloads)
    """
    total_workload = sum(client.affordable_workload for client in selected_clients)
    return total_workload

########################################
# Training LSTM with Failure History
########################################
def prepare_failure_data(history, sequence_length=1):
    sequences, labels = [], []
    for client_id, timestamps in history.items():
        print(f"Client ID: {client_id}, Failure Timestamps: {timestamps}")
        if len(timestamps) < sequence_length + 1:
            print(f"Client {client_id} does not have enough timestamps for training.")
            continue
        # Calculate time differences in seconds (timestamps should be in seconds)
        time_diffs = np.diff(timestamps).tolist()  # Time differences between consecutive failures in seconds
        for i in range(len(time_diffs) - sequence_length):
            sequences.append(time_diffs[i:i + sequence_length])
            labels.append(time_diffs[i + sequence_length])  # The next time difference
    if not sequences:
        return None, None
    sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # LSTM expects (batch_size, seq_len, features)
    labels = torch.tensor(labels, dtype=torch.float32)
    return sequences, labels


########################################
# Train LSTM to predict seconds until failure
########################################
def train_failure_predictor(failure_history, epochs=60, lr=0.001):
    X, y = prepare_failure_data(failure_history)

    if X is None:
        print("❌ Not enough history to train LSTM.")
        return None

    print(f"X: {X.shape}, y: {y.shape}")

    model = LSTMFailurePredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out.squeeze(), yb)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item():.4f}")

    return model


########################################
# Predict seconds until next failure
########################################
def predict_time_to_failure(model, failure_history, client_id):
    if not isinstance(failure_history, dict) or client_id not in failure_history:
        return None

    if model is None:
        print("⚠️ No trained LSTM model found. Using fallback prediction.")
        return 50.0  # <-- fallback prediction value (seconds)

    failure_timestamps = failure_history[client_id]

    # Case 1: Client has at least 2 failure timestamps → use their own intervals (in seconds)
    if len(failure_timestamps) >= 2:
        # Calculate time differences between failure timestamps (in seconds)
        intervals = np.diff(failure_timestamps).tolist()
        if len(intervals) >= 5:
            input_intervals = intervals[-5:]  # Use the last 5 intervals (in seconds)
        else:
            padding = [intervals[-1]] * (5 - len(intervals))
            input_intervals = padding + intervals  # Padding with the last interval if needed

    else:
        # Case 2: Client has <2 failures → use the average of other clients
        all_intervals = []
        for cid, timestamps in failure_history.items():
            if cid == client_id or len(timestamps) < 2:
                continue
            all_intervals.extend(np.diff(timestamps).tolist())  # Get time intervals in seconds

        if not all_intervals:
            # Cold-start fallback: just use 5 copies of a default interval (e.g., 50 seconds)
            input_intervals = [50] * 5
        else:
            avg_interval = sum(all_intervals) / len(all_intervals)  # Calculate average time interval in seconds
            input_intervals = [avg_interval] * 5

    # Prepare input for the LSTM model (convert to tensor format)
    x_input = torch.tensor(input_intervals, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # LSTM expects 3D input

    # Make prediction
    with torch.no_grad():
        pred = model(x_input).item()

    # Ensure prediction is non-negative
    return max(0.0, pred)

########################################
# Maximum Workload Based on Predicted Failure Time
########################################
def adjust_workload_based_on_failure(client, predict_time_to_failure, r1, r2, r3, total_workload, training_times, avg_epoch):
    """
    Adjust workload dynamically based on predicted failure time.
    If the failure time is predicted to be close, reduce workload; otherwise, increase.
    """


    # Calculate the average training time per epoch across clients
    avg_training_time = sum(training_times) / len(training_times)


    # Calculate the time per unit workload (seconds per unit of workload)
    time_per_unit_workload = avg_training_time / total_workload

    # Maximum workload based on predicted time to failure
    max_workload = predict_time_to_failure / time_per_unit_workload

    failure_ratio = predict_time_to_failure / np.mean(list(avg_epoch.values()))
    r3_max = max_workload
    r3 = r3_max * (1 - np.exp(-failure_ratio))

    # Define threshold failure as 10% of the predicted failure time
    threshold_percentage = 0.10  # For example, 10%
    threshold_failure = predict_time_to_failure * threshold_percentage

	    # State 1: Increase workload by r2 if affordable workload is less than max_workload and failure time is approaching
    if client.affordable_workload < max_workload and predict_time_to_failure < threshold_failure:
        client.affordable_workload += r2

    # State 2: Increase workload by r1 if we are not close to failure and in the middle of the workload range
    elif (predict_time_to_failure >= threshold_failure and
          client.affordable_workload < max_workload):
        client.affordable_workload += r1

    # State 3: Approaching failure and max_workload, increase workload by r3
    else:
        client.affordable_workload += r3

    # If we exceed max workload, drop the workload to zero
    if client.affordable_workload > max_workload:
        client.affordable_workload = 0

    return client.affordable_workload


########################################
# All Predicted failure time and steps together
########################################
def simulate_round(selected_clients, failure_history, r1, r2, model, round_number,failure_duration,r3):
    for client in selected_clients:


        # Predict the failure time
        predicted_failure_time = predict_time_to_failure(model, failure_history, client.cid)
        actual_failure_time = failure_duration  # This is the actual failure time
        error = abs(predicted_failure_time - actual_failure_time)


        # Adjust the client's workload dynamically based on failure prediction
        client.affordable_workload = adjust_workload_based_on_failure(client, predicted_failure_time, r1, r2, r3)

        # Update the task assignment stage (Increasing, Stable, Dropout)
        if predicted_failure_time <= 0:
            stage = "DROPOUT"
        elif client.affordable_workload > client.upper_bound:
            stage = "INCREASING"
        else:
            stage = "STABLE"

        # Adjust workload based on the stage
        client.affordable_workload = adjust_task_assignment(client, r1, r2, stage)

        # Log the updated data
        with open("client_failure_prediction.csv", "a") as log_file:
          log_file.write(f"{round_number},{client.cid},{client.affordable_workload:.2f},{predicted_failure_time:.2f},{actual_failure_time:.2f},{error:.2f}\n")



    # Calculate the total workload for this round
    total_workload = calculate_total_workload(selected_clients)
    print(f"Total Workload for Round {round_number}: {total_workload} units")



########################################
# Energy Consumption
########################################
def compute_energy(
    affordable_workload,
    model_size_bits,
    samples_per_epoch=100,
    train_time_sample=0.01,       # seconds per sample
    computation_power=0.5,        # Watts
    transmitter_power=0.1,        # Watts
    channel_capacity=1_000_000    # bits per second
):
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
        self.initial_affordable_workload = self.affordable_workload
        self.num_epochs = int(self.affordable_workload)
        self.lower_bound = 10
        self.upper_bound = 20
        self.threshold = 20
        self.failure_history = []





    def initialize_affordable_workload(self):
        """Generate a client's affordable workload using a normal distribution."""
        mu_k = np.random.uniform(50, 60)  # Mean workload
        sigma_k = np.random.uniform(mu_k / 4, mu_k / 2)  # Standard deviation
        value = max(0, np.random.normal(mu_k, sigma_k))
        print(f"Client {self.cid} initialized with workload: {value}")
        self.initial_affordable_workload = value
        return value  # Ensure workload is non-negative



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



        # ✅ Ensure client_id exists in client_history
        if client_id not in client_history:
            initial_epochs = int(self.initial_affordable_workload)
            print(f"⚠️ Warning: Client {client_id} not found in client_history. Initializing with default values.")
            client_history[client_id] = {
                "L": 10, "H": 20, "epochs": initial_epochs, "task": "Easy",
                "training_time": [], "accuracy": []
            }

        num_epochs = config.get("num_epochs")
        if num_epochs is None:
          num_epochs = int(self.initial_affordable_workload)  # Fallback to initial workload if not specified

        self.num_epochs = num_epochs
        training_epochs[client_id] = self.num_epochs


        print(f"Client {client_id}: Training for {num_epochs} epochs (Affordable Workload: {self.affordable_workload:.2f})")

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
        if "training_time" not in client_history[client_id]:
          client_history[client_id]["training_time"] = []
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

    log_file_path = "client_task_log.csv"
    client_history = {}
    clients = []

    for i in range(args['NUM_DEVICES']):
        client = FlowerClient(
            model=Net(),
            trainloader=trainloaders[i],
            testloader=testloader,
            valloader=valloaders[i],
            cid=i
        )
        clients.append(client)
        client_history[i] = {}
        client_id = client.cid

    

    # ✅ Initialize Failure Tracking (Only once)
    unavailability_tracker = {cid: 0 for cid in range(args['NUM_DEVICES'])}  # 0 = available, >0 = failure duration
    failure_log = []  # Store failures (Client ID, Failure Duration, Recovery Time)
    training_times = {}  # Store training times for each client
    failure_history = {}  # Store failure timestamps for each client
    client_to_server_map = {}
    model_size_bits = 1_000_000   # adjust based on your actual model size
    samples_per_epoch = 100       # adjust if you use different batch/sample sizes
    train_time_sample = 0.01      # seconds per sample
    r1 = args['base_r1']
    r2 = args['base_r2']
    r3 = args['base_r3']






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
    edge_servers, client_to_server_map = initialize_edge_servers_with_backups(edge_devices, args['NUM_EDGE_SERVERS'])
    assign_random_backups(edge_servers, edge_devices)
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
            log_file.write("Round,Client,Status,Duration,RecoveryTime,TrainingTime,LowerBound_before,UpperBound_before,LowerBound_after,UpperBound_after,ClientAffordableWorkload,State,NextPredictedFailure,ActualFailure,Error,energyCompConsumed, energyCommConsumed, TotalEnrgy,TotalTrainingTime \n")


    # ✅ Train LSTM model at the start
    lstm_model = train_failure_predictor(client_failure_history)

    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):
        start_time = time.time()

        # ✅ Select only available clients for training
        available_clients = [cid for cid in range(args["NUM_DEVICES"]) if unavailability_tracker[cid] == 0]
        if not available_clients:
            print(f"⚠️ No available clients in round {round_number}. Skipping round.")
            continue

        selected_clients = available_clients

        # ✅ Simulate failures before selecting clients
        failure_log, recovered_this_round= simulate_failures(args, unavailability_tracker, failure_log, round_number, training_times, selected_clients, lstm_model, failure_history)
       
        # Simulate failure per round
        if round_number == 1:
          failed_servers = set()
        simulate_edge_failures(edge_servers, failed_servers, failure_rate=0.05, client_to_server_map = client_to_server_map)
        # ✅ Handle edge server failures after simulating client failures
        for server in edge_servers:
          if server.is_failed():
            server.handle_failure(edge_servers, client_to_server_map)

        
        
        
        # ✅ Adjust workloads before selecting clients using failure predictions
        training_times = {client.cid: client_history.get(client.cid, {}).get("training_time", [0])[-1] for client in clients}
        training_epochs = {client.cid: client.num_epochs for client in clients}
        avg_epoch = calculate_avg_epochs_per_client(training_epochs, len(clients))
        print(f"Training epochs per client: {training_epochs}")  # ✅ Debug check
        print(f"Average epoch time per client: {avg_epoch}")


        predicted_failure_time = predict_time_to_failure(lstm_model, failure_history, client_id)
        L_tk_before, H_tk_before, L_tk_after, H_tk_after, stage = adjust_task_assignment(
                round_number=round_number,
                clients=clients,
                selected_clients=selected_clients,
                log_file_path=log_file_path,
                alpha=args['alpha'],
                r1=r1,
                r2=r2,
                training_times=training_times,  # ideally tracked over rounds
                predicted_failure_time=predicted_failure_time,
                failure_history=failure_history,
                lstm_model=lstm_model,
                num_epochs=training_epochs,
                avg_epoch=avg_epoch,
                args=args
            )


        for client_id in selected_clients:

            client = clients[client_id]  # ✅ Correctly reference the client
            

            AffordableWorkloadInitilize = client.affordable_workload
            num_epochs = compute_training_rounds(client_id, clients, args['base_k1'])
            client = clients[client_id]  # ✅ Correctly reference the client

            # ✅ Train the client before logging
            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})
            training_time = train_metrics["training_time"]
            training_times[client_id] = training_time
            total_training_time = sum(training_times.values())

            # Compute energy
            affordable_workload = client.affordable_workload
            energyCompConsumed, energyCommConsumed, total_energy_consumed, training_time = compute_energy(
              affordable_workload=affordable_workload,
              model_size_bits=model_size_bits,
              samples_per_epoch=samples_per_epoch,
              train_time_sample=train_time_sample
              )


            predicted_time = getattr(client, "predicted_failure_time", -1.0)




            # ✅ Log training clients with workload details
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{round_number},{client_id},TRAINING,,0,0,{training_time:.2f},{L_tk_before},{H_tk_before},{L_tk_after},{H_tk_after},{client.affordable_workload:.2f},{num_epochs},{stage},0, {energyCompConsumed}, {energyCommConsumed}, {total_energy_consumed}, {total_training_time}\n")

            with open("client_failure_prediction.csv", "a") as log_file:
                log_file.write(f"{round_number},{client.cid},{predicted_time:.2f}\n")


        # ✅ Edge Aggregation
        if round_number % args["k2"] == 0:
            print(f"🔹 Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
               if not edge_server.is_failed():
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
        'FAILURE_RATE': 0.2,
        'FAILURE_DURATION': 50,
        'alpha': 0.95,
        'base_r1': 0.2,
        'base_r2': 0.1,
        'base_r3': 0.8,
        'base_k1': 60,
        'NUM_EPOCHS': 60,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()



