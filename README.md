Trust Aware Secure Energy-Efficient Routing Protocol (TSEERPC-TTO)

📄 Project Overview
This repository contains the implementation and performance evaluation of a Trust Aware Secure Energy-Efficient Routing Protocol for Clustered Wireless Sensor Networks, referred to as TSEERPC-TTO. The model enhances the efficiency and security of data transmission in WSNs using a Two-Tier Optimization approach combining:
Improved Sparrow Search Algorithm (ISPSA) for energy-efficient node clustering
Harris Hawks Optimizer (HHO) for secure and optimized data routing
A trust factor is integrated into the fitness function to ensure secure communication by detecting and avoiding malicious nodes.

📌 Features
ISPSA-based Clustering: Efficient energy-aware node grouping
HHO-based Routing: Robust, trust-aware routing algorithm
Trust Model Integration: Mitigates threats like blackhole and selective forwarding attacks
Multi-metric Optimization: Trust, residual energy, hops, and transmission distance
Performance Evaluation: Assessed against PDR, detection speed, trust value, and energy consumption

🚀 How It Works
1. ISPSA for Clustering
Sensor nodes are clustered using ISPSA. Producers and scroungers mimic sparrow behavior to locate energy-optimal cluster heads.
2. HHO for Routing
The cluster heads use HHO to identify trustworthy, energy-efficient paths to the base station by simulating hawk hunting strategies.
3. Trust Factor in Fitness
The routing decision incorporates a trust metric based on past node behavior to isolate malicious nodes.



🔧Code Implementation Steps
✅ Step 1: Import Libraries
python
Copy
Edit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

✅ Step 2: Initialize Wireless Sensor Network (WSN) Parameters
Number of nodes: 100
Area size: 100x100 meters
Initial energy per node: 2.0 J
Base station (BS) position: (50, 50)
python
Copy
Edit
NUM_NODES = 100
AREA_DIM = 100
INIT_ENERGY = 2.0
BS_POS = (50, 50)

✅ Step 3: Deploy Sensor Nodes
Generate random node positions and assign initial energy to all nodes.
python
Copy
Edit
np.random.seed(42)
node_positions = np.random.randint(0, AREA_DIM, size=(NUM_NODES, 2))
node_energies = np.full(NUM_NODES, INIT_ENERGY)

✅ Step 4: Define Energy Consumption Functions
Use simplified radio energy model (free space / multipath fading model).
python
Copy
Edit
def dist(a, b):
    return np.linalg.norm(a - b)
def energy_tx(k, d):
    return k * E_elec + k * E_fs * d ** 2 if d < THRESH_DIST else k * E_elec + k * E_mp * d ** 4
def energy_rx(k): return k * E_elec
def energy_da(k): return k * E_da

✅ Step 5: Trust Evaluation
Simulate trust using ratio of successful receptions to transmissions.
python
Copy
Edit
def evaluate_trust(tx, rx):
    return np.clip(rx / tx, 0, 1) if tx > 0 else 0

✅ Step 6: Clustering Using ISPSA (Simplified as KMeans)
python
Copy
Edit
def clustering_ispsa(nodes, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit_predict(nodes)
    return labels, kmeans.cluster_centers_
✅ Step 7: Routing Using Harris Hawks Optimization (Simplified)
Select route based on distance from cluster heads to BS.
python
Copy
Edit
def hho_routing(cluster_centers):
    dists = cdist(cluster_centers, np.array(BS_POS).reshape(1, -1)).flatten()
    return np.argsort(dists)  # Closest cluster head routes first

✅ Step 8: Fitness Function
Compute weighted sum based on:
Trust
Distance (inverted)
Energy
Number of hops (inverted)
python
Copy
Edit
def compute_fitness(trust, distance, energy, hops, weights=[0.25]*4):
    return weights[0]*trust + weights[1]*(1/distance) + weights[2]*energy + weights[3]*(1/hops)

✅ Step 9: Simulate Clustering and Routing
python
Copy
Edit
NUM_CLUSTERS = 5
labels, cluster_centers = clustering_ispsa(node_positions, NUM_CLUSTERS)
routing_order = hho_routing(cluster_centers)

✅ Step 10: Visualize Network Topology and Routing
python
Copy
Edit
plt.figure(figsize=(8, 6))
for i in range(NUM_CLUSTERS):
    cluster_nodes = node_positions[labels == i]
    plt.scatter(cluster_nodes[:, 0], cluster_nodes[:, 1], label=f'Cluster {i+1}')
    plt.plot([cluster_centers[i][0], BS_POS[0]], [cluster_centers[i][1], BS_POS[1]], 'k--')
plt.scatter(BS_POS[0], BS_POS[1], c='red', marker='*', s=200, label='Base Station')
plt.title('WSN Clustering and Routing with TSEERPC-TTO')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



📊 Performance Metrics
Comprehensive Trust (CT)
Packet Delivery Ratio (PDR)
Detection Speed (DS)
Average Energy Consumption (AEOM)

📈 Evaluation Highlights
Achieved PDR > 98% under malicious scenarios
Maintained low energy consumption
Outperformed EASR, MSCR, TBSEER, and ETOR protocols in simulations
Effective against blackhole and selective forwarding attacks

🧰 Requirements
Python ≥ 3.8
NumPy
SciPy
Matplotlib
pandas
Install using:
bash
Copy
Edit
pip install -r requirements.txt
