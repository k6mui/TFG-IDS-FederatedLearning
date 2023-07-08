# TFG-IDS-FederatedLearning

## 1. Introduction
This repository contains the necessary code to conduct the Final Degree Project in Telecommunications Engineering at the Polytechnic University of Madrid. This project performs an evaluation of Federated Learning applied to Intrusion Detection Systems (IDS) in a cross-silo configuration for heterogeneous networks.

## 2. Installation "Setup"
To install the required dependencies, run the following command:
```pip install -r requirements.txt```

## 3. Project Preparation
Go to the `./datasets` directory and follow the instructions provided there to acquire the project's starting datasets: NF-UNSW-NB15-v2, NF-ToN-IoT-v2, and NF-CSE-CIC-IDS2018-v2.

## 4. Instructions for Launching the Project on the Cloud
1. Go to the following link: https://console.cloud.google.com/. Any type of cloud machine is valid to execute the project: EC2, Azure, etc.
2. In the "VM Instances" section, select "Create Instance".
3. Choose an e2-standard-2 machine.
4. Select "Ubuntu 20.04" x86-64 for boot disk.
5. Create the instance.
6. Go to "Configure Firewall Rules".
7. Add two rules that allow both incoming and outgoing traffic for the TCP protocol on port 4687 (or any other port where you wish to launch the server).
8. Install the project repository on the virtual machine: git clone https://github.com/k6mui/TFG-IDS-FederatedLearning.git
9. Modify the `server.py` file, entering the internal IP of the virtual machine in the "serverAdress" variable.
10. Repeat steps 8 and 9 on the machine where you wish to launch the clients.
11. Enter the external IP of the virtual machine in the "server_address" variable.

## 6. Launch the Project
To launch the project, you will need to run the `server.py` file in the cloud, within the directory with the name of the model you want to launch, and do the same with the `client.py` file on the machine. The project is set up so that training does not start until three instances of `client.py` have been launched.

The command has a parameter that can be '1', '2' or '3' depending on the client you want to launch, as each one will represent one of the different datasets. The command should be in the form of `python3 client.py 1`, `python3 client.py 2` or `python3 client.py 3`. You can launch any combination of clients, and you can launch the same one repeatedly.
Run the following commands as per your needs:




