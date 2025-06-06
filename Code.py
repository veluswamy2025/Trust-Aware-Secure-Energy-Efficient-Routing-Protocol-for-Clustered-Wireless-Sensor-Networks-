#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# =============================
# TSEERPC-TTO Model for WSN
# =============================

# Basic Network Setup
NUM_NODES = 100
AREA_DIM = 100  # 100x100 meters
INIT_ENERGY = 2.0  # in Joules
BS_POS = (50, 50)

np.random.seed(42)
node_positions = np.random.randint(0, AREA_DIM, size=(NUM_NODES, 2))
node_energies = np.full(NUM_NODES, INIT_ENERGY)

# Parameters for energy model
E_elec = 50e-9
E_fs = 10e-12
E_mp = 0.0013e-12
E_da = 5e-9
THRESH_DIST = np.sqrt(E_fs / E_mp)

def dist(a, b):
    return np.linalg.norm(a - b)

def energy_tx(k, d):
    if d < THRESH_DIST:
        return k * E_elec + k * E_fs * d ** 2
    else:
        return k * E_elec + k * E_mp * d ** 4

def energy_rx(k):
    return k * E_elec

def energy_da(k):
    return k * E_da

# Placeholder Trust Evaluation
def evaluate_trust(tx, rx):
    return np.clip(rx / tx, 0, 1) if tx > 0 else 0

# Clustering Phase using ISPSA (simplified)
def clustering_ispsa(nodes, num_clusters):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    labels = kmeans.fit_predict(nodes)
    return labels, kmeans.cluster_centers_

# Routing Phase using HHO (simplified)
def hho_routing(cluster_centers):
    from scipy.spatial.distance import cdist
    bs_vector = np.array(BS_POS).reshape(1, -1)
    dists = cdist(cluster_centers, bs_vector).flatten()
    best_route = np.argsort(dists)  # Closest first
    return best_route

# Fitness Function
def compute_fitness(trust, distance, energy, hops, weights=[0.25]*4):
    return weights[0]*trust + weights[1]*(1/distance) + weights[2]*energy + weights[3]*(1/hops)

# Simulation Start
NUM_CLUSTERS = 5
labels, cluster_centers = clustering_ispsa(node_positions, NUM_CLUSTERS)
routing_order = hho_routing(cluster_centers)

# Visualization
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

