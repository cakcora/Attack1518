import pandas as pd
import itertools
import random
import torch
import numpy as np
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GAT
from deeprobust.graph.targeted_attack import RND
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.targeted_attack import IGAttack

import networkx as nx

# Create a sample graph
#G = nx.Graph()
#G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6)])



T_A_O=['1518'] #, '1922', '1519', '1819']

store= []

#Target_alom_orbit = '1518'
#Connecting_alom_orbit='1518'

keepvalue= np.zeros((6, 10))

filename='New_idea_Add_LS_0_1518.csv'

for tao in T_A_O:
    Target_alom_orbit = tao
    Connecting_alom_orbit='1519'
    alom_label = 6


    with open('C:\pythonProject1\orca-py\mybao\dpr_cora.out', 'r') as file:
        lines = file.readlines()
    ####### Read the data from the file
    # with open('C:\pythonProject1\dpr_cora_4_graphlet.out', 'r') as file:
    # lines = file.readlines()
    # Process the lines to create a list of lists
    data_list = [list(map(int, line.split())) for line in lines]

    # Convert the list of lists to a NumPy array

    graphlet_features = np.array(data_list)
    # print(graphlet_features)

    mylist = []
    for i in range(len(graphlet_features)):
        arr = graphlet_features[i]
        # print(arr)
        ###########  sorted_arr = np.sort(arr, axis=0)[::-1]
        sorted_indices = np.argsort(arr)[::-1]
        # print(sorted_indices)
        ###########    mylist.append([i, sorted_indices[0], sorted_indices[1]])

        if sorted_indices[0] < sorted_indices[1]:
            s1 = str(sorted_indices[0]) + str(sorted_indices[1])
        else:
            s1 = str(sorted_indices[1]) + str(sorted_indices[0])

        ############  print(s1)  ############

        mylist.append([i, sorted_indices[0], sorted_indices[1], s1])

    my_array = np.array(mylist)

    df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])

    #print(df_2d)
    #print(len(df_2d))

    #########################

    import networkx as nx

    ### Create an empty graph
    G = nx.Graph()

    #### Open the text file for reading
    with open('graph_cora_edge_list.txt', 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into nodes (assuming space or tab separation)
            nodes = line.strip().split()

            # Convert nodes to integers (if needed)
            nodes = [int(node) for node in nodes]

            # Add the edge to the graph
            G.add_edge(*nodes)

    ################# Original Graph Degree ##############

    data = Dataset(root='C:\pythonProject1', name='cora')
    adj, features, labels = data.adj, data.features, data.labels

    keys = list(range(len(labels)))
    # Convert the NumPy array to a Python list
    values = labels.tolist()
    my_dict = {k: v for k, v in zip(keys, values)}

    # Add node labels to the existing graph 'G'
    node_labels = my_dict  # {1: "Node A", 2: "Node B", 3: "Node C", 4: "Node D"}

    # Assign node attributes for labels
    nx.set_node_attributes(G, node_labels, "label")

    # Access node labels
    labels = nx.get_node_attributes(G, "label")

    print(my_dict)
    print(labels)

    same_label_node = [k for k, v in my_dict.items() if v == alom_label]
    print('same_label_node')
    print(same_label_node)

    two_orbit = df_2d['two_Orbit_type'].tolist()
    my_new_dict = {k: v for k, v in zip(keys, two_orbit)}

    print(my_new_dict)

    ######### Target Node list############
    same_orbit_type_node = [k for k, v in my_new_dict.items() if v == Target_alom_orbit]
    print('same_orbit_type_node')
    print(same_orbit_type_node)


    ########### select 20 target node from same label nodes, regardless orbit types ########

    target_node_20= same_label_node[0:40]
    same_label_node_rest=same_label_node[40:]

    print(target_node_20)
    print(same_label_node_rest)

    ######### Getting LS before adding ############

    shortest_paths = {}

    for i in range(len(target_node_20)):
        for j in range(len(same_label_node_rest)):
            source_node = target_node_20[i]
            target_node = same_label_node_rest[j]
            shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)
            shortest_paths[(source_node, target_node)] = shortest_path_length

    ##################### Getting score for avarage #####################

    # Calculate the sum of values
    total_sum = sum(shortest_paths.values())

    # Calculate the average
    average_before = total_sum / len(shortest_paths)

    # Print the average
    print("LS Score before adding ")
    print(average_before)

    #print(shortest_paths)


    ############ Getting LD score ###############
    ########### select 20 target node from same label nodes, regardless orbit types ########

    target_node_20 = same_label_node[0:40]

    all_nodes=list(my_dict.keys())

    #print(all_nodes)

    not_same_label_node=  [x for x in all_nodes if x not in same_label_node]

    #print(target_node_20)
    #print(not_same_label_node)

    ######### Getting LD before adding ############

    shortest_paths = {}

    for i in range(len(target_node_20)):
        for j in range(len(not_same_label_node)):
            source_node = target_node_20[i]
            target_node = not_same_label_node[j]
            shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)
            shortest_paths[(source_node, target_node)] = shortest_path_length

    ##################### Getting score for avarage #####################

    # Calculate the sum of values
    total_sum = sum(shortest_paths.values())

    # Calculate the average
    average_before = total_sum / len(shortest_paths)

    # Print the average
    print("LD Score before adding ")
    print(average_before)



    ################################# imediate neighbors and k hope neighbors
    a_k_hop=[]

    for i in target_node_20:
        target_node = i

            # Find immediate neighbors of the target node
        immediate_neighbors = list(G.neighbors(target_node))

            # Find two-hop neighbors using the neighbors function
        two_hop_neighbors = set()
        for neighbor in immediate_neighbors:
            two_hop_neighbors.update(G.neighbors(neighbor))

            # Remove the original target node from the two-hop neighbors set
        two_hop_neighbors.discard(target_node)

        immediate_neighbors.extend(list(two_hop_neighbors))

        a_k_hop.append(len(immediate_neighbors))
        #print(len(immediate_neighbors))

    average_a_k_hop = sum(a_k_hop) / len(a_k_hop)
    print("All k hop neighobrs before adding any edges")
    print(average_a_k_hop)

    ################ Task addition #################

    Target_node_list = target_node_20

    ######### Connecting Node list ############

    Connecting_orbit_type_node = [k for k, v in my_new_dict.items() if v == Connecting_alom_orbit]
    #print(Connecting_orbit_type_node)

    Connecting_common_elements = np.intersect1d(same_label_node, Connecting_orbit_type_node)
    #print(Connecting_common_elements)
    #print(len(Connecting_common_elements))

    # tnode = random.sample(list(idx_unlabeled), 20)
    # degrees = adj.sum(0).A1
    # print(degrees[0])

    degrees = adj.sum(0).A1

    tnode = []
    # n_perturbations=4

    for i in Connecting_common_elements:
        if int(degrees[i]) > 0:
            tnode.append(i)

    if len(tnode) > 5:
        tnode = tnode[0]
        # tnode = random.sample(list(tnode), 10)

    connecting_node_list = tnode
    print(tnode)
    print(labels[tnode])

    average_after_all_target = []
    average_a_k_hop_after_final2=[]

    for i in Target_node_list:
        target = i
        average_after = []
        average_a_k_hop_after=[]

        ######## this list sotre the short distance after each connection for a target node
        #for j in connecting_node_list:
        connecting = tnode
        if target == connecting:
            continue
        # print(i)
        # print(j)
        G.add_edge(target, connecting)

        ################
        ################################# imediate neighbors and k hope neighbors
        a_k_hop = []

        for i in target_node_20:
            target_node = i

            # Find immediate neighbors of the target node
            immediate_neighbors = list(G.neighbors(target_node))

            # Find two-hop neighbors using the neighbors function
            two_hop_neighbors = set()
            for neighbor in immediate_neighbors:
                two_hop_neighbors.update(G.neighbors(neighbor))

                # Remove the original target node from the two-hop neighbors set
            two_hop_neighbors.discard(target_node)

            immediate_neighbors.extend(list(two_hop_neighbors))

            a_k_hop.append(len(immediate_neighbors))
            # print(len(immediate_neighbors))

        average_a_k_hop = sum(a_k_hop) / len(a_k_hop)
        #print("All k hop neighobrs before adding any edges")
        #print(average_a_k_hop)
        average_a_k_hop_after.append(average_a_k_hop)


        ###############

        ######### Getting LS before adding ############

        shortest_paths = {}

        for i in range(len(target_node_20)):
            for j in range(len(same_label_node_rest)):
                source_node = target_node_20[i]
                target_node = same_label_node_rest[j]
                shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)
                shortest_paths[(source_node, target_node)] = shortest_path_length

        average = sum(shortest_paths.values()) / len(shortest_paths)
        average_after.append(average)
        # G.add_edges_from(edge_to_remove)
        # Add an edge between nodes
        edge_to_remove = (target, connecting)
        # print(edge_to_remove)
        G.remove_edge(*edge_to_remove)
        is_connected = nx.is_connected(G)
        # print(is_connected)
        if is_connected == 0:
            G.add_edge(target, connecting)

        ########### new here ########

        average_all_connection = sum(average_after) / len(average_after)
        # print(average_all_connection)

        average_after_all_target.append(average_all_connection)
        average_after_addition = sum(average_after_all_target) / len(average_after_all_target)
        # print(average_after_all_target)

        average_a_k_hop_after_final=sum(average_a_k_hop_after)/len(average_a_k_hop_after)
        average_a_k_hop_after_final2.append(average_a_k_hop_after_final)
        all_k_hop = sum(average_a_k_hop_after_final2) / len(average_a_k_hop_after_final2)

    # Print the average
    # print("Score before adding edges")
    # print(average_before)

    # print(len(average_after))
    print("LS Score after addition edges")
    print(average_after_addition)

    print("all k hope Score after addition LS edges ")
    print(all_k_hop)

    ################ Task addition II #################

    Target_node_list = target_node_20

    ######### Connecting Node list ############

    Connecting_orbit_type_node = [k for k, v in my_new_dict.items() if v == Connecting_alom_orbit]
    #print(Connecting_orbit_type_node)

    Connecting_common_elements = np.intersect1d(same_label_node, Connecting_orbit_type_node)
    #print(Connecting_common_elements)
    #print(len(Connecting_common_elements))

    # tnode = random.sample(list(idx_unlabeled), 20)
    # degrees = adj.sum(0).A1
    # print(degrees[0])

    degrees = adj.sum(0).A1

    tnode = []
    # n_perturbations=4

    for i in Connecting_common_elements:
        if int(degrees[i]) > 0:
            tnode.append(i)

    if len(tnode) > 5:
        tnode = tnode[0]
        # tnode = random.sample(list(tnode), 10)

    connecting_node_list = tnode
    print(tnode)
    print(labels[tnode])

    average_after_all_target = []
    average_a_k_hop_after_final2 = []

    for i in Target_node_list:
        target = i
        average_after = []
        average_a_k_hop_after = []
        ######## this list sotre the short distance after each connection for a target node
        #for j in connecting_node_list:
        connecting = tnode
        if target == connecting:
            continue
        # print(i)
        # print(j)
        G.add_edge(target, connecting)

        ################################# imediate neighbors and k hope neighbors
        a_k_hop = []

        for i in target_node_20:
            target_node = i

            # Find immediate neighbors of the target node
            immediate_neighbors = list(G.neighbors(target_node))

            # Find two-hop neighbors using the neighbors function
            two_hop_neighbors = set()
            for neighbor in immediate_neighbors:
                two_hop_neighbors.update(G.neighbors(neighbor))

                # Remove the original target node from the two-hop neighbors set
            two_hop_neighbors.discard(target_node)

            immediate_neighbors.extend(list(two_hop_neighbors))

            a_k_hop.append(len(immediate_neighbors))
            # print(len(immediate_neighbors))

        average_a_k_hop = sum(a_k_hop) / len(a_k_hop)
        # print("All k hop neighobrs before adding any edges")
        # print(average_a_k_hop)
        average_a_k_hop_after.append(average_a_k_hop)

        ######### Getting LD before adding ###########

        shortest_paths = {}

        for i in range(len(target_node_20)):
            for j in range(len(not_same_label_node)):
                source_node = target_node_20[i]
                target_node = not_same_label_node[j]
                shortest_path_length = nx.shortest_path_length(G, source=source_node, target=target_node)
                shortest_paths[(source_node, target_node)] = shortest_path_length

        average = sum(shortest_paths.values()) / len(shortest_paths)
        average_after.append(average)
        # G.add_edges_from(edge_to_remove)
        # Add an edge between nodes
        edge_to_remove = (target, connecting)
        # print(edge_to_remove)
        G.remove_edge(*edge_to_remove)
        is_connected = nx.is_connected(G)
        # print(is_connected)
        if is_connected == 0:
            G.add_edge(target, connecting)

        average_all_connection = sum(average_after) / len(average_after)
        # print(average_all_connection)

        average_after_all_target.append(average_all_connection)
        average_after_addition = sum(average_after_all_target) / len(average_after_all_target)
        # print(average_after_all_target)

        average_a_k_hop_after_final = sum(average_a_k_hop_after) / len(average_a_k_hop_after)
        average_a_k_hop_after_final2.append(average_a_k_hop_after_final)
        all_k_hop = sum(average_a_k_hop_after_final2) / len(average_a_k_hop_after_final2)

    # Print the average
    # print("Score before adding edges")
    # print(average_before)

    # print(len(average_after))
    print("LD Score after addition edges")
    print(average_after_addition)

    print("all k hope Score after addition LD edges ")
    print(all_k_hop)
    # Calculate the average
    # print("Difference")
    # print(average_after_addition - average_before)

    #store.append(tao)
    #store.append(len(average_after_all_target))
    #store.append(average_before)
    #store.append(average_after_addition)
