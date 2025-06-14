#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

# ----------- Parameters -------------
num_nodes = 100
area_size = (100, 100)
num_clusters = 5
iterations = 100
initial_energy = 2.0

# ----------- Node Initialization -------------
class Node:
    def __init__(self, node_id, position):
        self.id = node_id
        self.pos = position
        self.energy = initial_energy
        self.trust = np.random.uniform(0.8, 1.0)  # initial trust
        self.is_CH = False

nodes = [Node(i, np.random.rand(2) * area_size[0]) for i in range(num_nodes)]
base_station = np.array([50, 50])

# ----------- Trust Evaluation -------------
def compute_trust(transmitted, received):
    return received / transmitted if transmitted != 0 else 0

# ----------- Fitness Function -------------
def compute_fitness(node, bs_pos, cluster_size_weight=0.25, trust_weight=0.25, energy_weight=0.25, distance_weight=0.25):
    distance = np.linalg.norm(node.pos - bs_pos)
    trust = node.trust
    energy = node.energy
    num_hops = 1  # simplified
    # Normalize values (assume max energy 2.0, max distance 150)
    fitness = (cluster_size_weight * (1/num_hops) +
               trust_weight * trust +
               energy_weight * (energy / 2.0) +
               distance_weight * (1 - distance / 150))
    return fitness

# ----------- ISPSA for Clustering -------------
def ispsa_clustering(nodes):
    # Simulate selection of CHs based on energy + trust
    CHs = sorted(nodes, key=lambda n: compute_fitness(n, base_station), reverse=True)[:num_clusters]
    for ch in CHs:
        ch.is_CH = True
    return CHs

# ----------- HHO for Routing (Simplified) -------------
def hho_routing(CHs, nodes):
    for node in nodes:
        if not node.is_CH:
            dists = [np.linalg.norm(node.pos - ch.pos) for ch in CHs]
            closest_ch = CHs[np.argmin(dists)]
            # Assign node to closest CH
            node.cluster = closest_ch.id

# ----------- Main Simulation -------------
for it in range(iterations):
    print(f"Iteration {it+1}")
    CHs = ispsa_clustering(nodes)
    hho_routing(CHs, nodes)

    # Energy update example (simplified)
    for node in nodes:
        if node.is_CH:
            node.energy -= 0.05
        else:
            node.energy -= 0.01

    # Trust re-evaluation (simplified)
    for node in nodes:
        node.trust = compute_trust(transmitted=100, received=95 + np.random.randint(-5, 5))

# ----------- Result Output -------------
print("Final Energy and Trust of CHs:")
for ch in [n for n in nodes if n.is_CH]:
    print(f"Node {ch.id} - Energy: {ch.energy:.2f}, Trust: {ch.trust:.2f}")

