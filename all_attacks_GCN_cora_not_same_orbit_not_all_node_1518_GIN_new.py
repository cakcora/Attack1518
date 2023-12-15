import pandas as pd
import itertools
import torch
import numpy as np
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.defense import GAT
from deeprobust.graph.targeted_attack import RND
from deeprobust.graph.targeted_attack import FGA
from deeprobust.graph.targeted_attack import IGAttack
from GIN import GIN

# Read the data from the file
#with open('C:\pythonProject1\orca-py\dpr_cora.out', 'r') as file:
    #lines = file.readlines()
"""
with open('C:\pythonProject1\orca-py\dpr_cora.out', 'r') as file:
    lines = file.readlines()
# Read the data from the file
#with open('C:\pythonProject1\dpr_cora_4_graphlet.out', 'r') as file:
    #lines = file.readlines()
# Process the lines to create a list of lists
data_list = [list(map(int, line.split())) for line in lines]

# Convert the list of lists to a NumPy array

graphlet_features = np.array(data_list)
#print(graphlet_features)


mylist = [ ]
for i in range(len(graphlet_features)):
  arr=graphlet_features[i]
  #sorted_arr = np.sort(arr, axis=0)[::-1]
  sorted_indices = np.argsort(arr)[::-1]
  mylist.append([i, sorted_indices[0]])


my_array = np.array(mylist)
df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type'])
print(len(df_2d))
"""

# Read the data from the file
#with open('C:\pythonProject1\orca-py\dpr_cora.out', 'r') as file:
    #lines = file.readlines()

with open('C:\pythonProject1\orca-py\dpr_cora.out', 'r') as file:
    lines = file.readlines()
####### Read the data from the file
#with open('C:\pythonProject1\dpr_cora_4_graphlet.out', 'r') as file:
    #lines = file.readlines()
# Process the lines to create a list of lists
data_list = [list(map(int, line.split())) for line in lines]

# Convert the list of lists to a NumPy array

graphlet_features = np.array(data_list)
print(graphlet_features)


mylist = []
for i in range(len(graphlet_features)):
    arr = graphlet_features[i]
    print(arr)
  ###########  sorted_arr = np.sort(arr, axis=0)[::-1]
    sorted_indices = np.argsort(arr)[::-1]
    print(sorted_indices)
  ###########    mylist.append([i, sorted_indices[0], sorted_indices[1]])

    if sorted_indices[0] < sorted_indices[1]:
        s1 = str(sorted_indices[0]) + str(sorted_indices[1])
    else:
        s1 = str(sorted_indices[1]) + str(sorted_indices[0])

    ############  print(s1)  ############

    mylist.append([i, sorted_indices[0], sorted_indices[1], s1])

my_array = np.array(mylist)
df_2d = pd.DataFrame(my_array, columns=['node_number', 'Orbit_type_I', 'Orbit_type_II', 'two_Orbit_type'])
print(df_2d)

#df = pd.DataFrame(df_2d)
# Save the DataFrame as a CSV file
#df.to_csv('same_orbit_miss.csv', index=False)
#df.to_csv('cora_orbit_check.csv', index=False)

#df_2d = df_2d.iloc[0:2110, :]

#df_2d = df_2d.iloc[0:2485, :]
#print(df_2d)

#value = df_2d.iloc[3, 1]  # Row index 1 and column index 2 (0-based index)
#print(value)

####################

import itertools
import numpy
import networkx as nx
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from tqdm import tqdm
from net import Nettack1 ####### our modified code
from nettackonlyadd import NettackAdd  ######for addition only
import time
import torch
import numpy as np
import torch.nn.functional as F
from deeprobust.graph import utils
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from tqdm import tqdm
import random
import math
import pandas as pd
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from numba import jit
from allfnc import get_linearized_weight, normalize_adj, compute_logits, find_all_2hop_neighbors1,select_lists_with_element1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def select_nodes(target_gcn=None):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''

    if target_gcn is None:
        target_gcn = GCN(nfeat=features.shape[1],
                  nhid=16,
                  nclass=labels.max().item() + 1,
                  dropout=0.5, device=device)
        target_gcn = target_gcn.to(device)
        target_gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    target_gcn.eval()
    output = target_gcn.predict()

    margin_dict = {}
    for idx in idx_test:
        margin = classification_margin(output[idx], labels[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    high = [x for x, y in sorted_margins[: 10]]
    low = [x for x, y in sorted_margins[-10: ]]
    other = [x for x, y in sorted_margins[10: -10]]
    other = np.random.choice(other, 20, replace=False).tolist()

    return high + low + other

"""
def test1(adj, features, target_node):
    ''' test on GCN '''
    gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gcn.eval()
    output = gcn.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    print(output.argmax(1)[target_node])
    print(labels[target_node])
    #acc_test = accuracy(output[idx_test], labels[idx_test])

    #print("Overall test set results:",
     #     "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()
"""

def test_GAT(adj, data, target_node):
    ''' test on GAT '''
    pyg_data = Dpr2Pyg(data)
    gat = GIN(nfeat=features.shape[1], nhid=8, heads=8, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    gat = gat.to(device)

    perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
    pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation
    gat.fit(pyg_data, verbose=False)  ############ train with earlystopping

    #gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device)
    #gcn = gcn.to(device)
    #gcn.fit(features, adj, labels, idx_train, idx_val, patience=30)
    gat.eval()
    output = gat.predict()
    probs = torch.exp(output[[target_node]])[0]
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (output.argmax(1)[target_node] == labels[target_node])
    print('Target node probs: {}'.format(probs.detach().cpu().numpy()))
    print(output.argmax(1)[target_node])
    print(labels[target_node])
    #acc_test = accuracy(output[idx_test], labels[idx_test])

    #print("Overall test set results:",
     #     "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()





#data = Dataset(root='C:\pythonProject1', name='cora') #Dataset(root='/tmp/', name=args.dataset)
data = Dataset(root='C:\pythonProject1', name='cora')
adj, features, labels = data.adj, data.features, data.labels

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled_all = np.union1d(idx_unlabeled, idx_train)

######################### Setup Surrogate model  #########################

surrogate = GIN(nfeat=features.shape[1], nclass=labels.max().item()+1,nhid=16, dropout=0, device=device)
surrogate = surrogate.to(device)
pyg_data = Dpr2Pyg(data)
#perturbed_adj = adj.tocsr()           ###########   perturbed_data.adj
#pyg_data.update_edge_index(perturbed_adj)  ############ inplace operation


surrogate.fit(pyg_data, verbose=False)  ############ train with earlystopping
#surrogate.fit(features, adj, labels, idx_train, idx_val)
nclass=labels.max().item()+1

#tnode = random.sample(list(idx_unlabeled), 20)
#degrees = adj.sum(0).A1
#print(degrees[0])

degrees = adj.sum(0).A1
#tnode=[]
#n_perturbations=4

#for i in idx_unlabeled:
    #if int(degrees[i]) >= n_perturbations:
        #tnode.append(i)

#filenames = ['file11.csv', 'file21.csv', 'file31.csv', 'file41.csv', 'file51.csv']

filenames = ['new_GIN_Cora_all.csv']
# Assuming you have a loop where you generate data for each CSV file
for filename in filenames:

    degrees = adj.sum(0).A1

    node_list = select_nodes()
    num = len(node_list)

    #tnode = []

    #for i in idx_unlabeled:
        #if int(degrees[i]) > 1:
            #tnode.append(i)

    #node_list = random.sample(list(idx_unlabeled), 50)

    #node_list = random.sample(tnode, 10)

    #num = len(node_list)

    print(node_list)
    rmv_del_edge=[]
    ###########  finding less node degree #######

    miss= np.zeros((9, 5))
    #edgemod= [1]

    edgemod= [1] #, 2, 3, 4, 5]

    for gp in range(len(edgemod)):
        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0


        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(adj).tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = utils.to_scipy(features).tolil()
            modified_features = utils.to_scipy(features).tolil()
        else:
            ori_adj = adj.tolil()
            modified_adj = adj.tolil()
            lst_modified_adj = adj.tolil()  ########## for comparison only
            work_modified_adj = adj.tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = features.tolil()
            modified_features = features.tolil()


        for target_node in tqdm(node_list):
            ############ Define the column name and the target value ########
            column_name = 'two_Orbit_type'
            Target_node = target_node
            Target_node_orbit = df_2d.iloc[Target_node, 3]
            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == Target_node_orbit].tolist()
            perterbation_list = [[Target_node, value] for value in matching_indices]

            #print(perterbation_list)

            #potential_edges = np.column_stack((np.tile(target_node, adj.shape[0] - 1), np.setdiff1d(np.arange(adj.shape[0]), target_node)))
            #perterbation_list = potential_edges.astype("int32")
            #perterbation_list =perterbation_list.tolist()

            print(perterbation_list)
            print(len(perterbation_list))

            #nofnode_modification_lst = [1]
            #nofnode_modification = nofnode_modification_lst[0]

            ########################### Using edge one by one  ##############################
            degrees = adj.sum(0).A1
            print(degrees[target_node])
            #if budget >= 5:
                #n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            #else:
            n_perturbations= budget #int(degrees[target_node])

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                print(edge_with_score)

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])  # reverse=True)

                print(sorted_list)

                print(len(sorted_list))

                print(sorted_list[0])

                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]]


                lst_modified_adj = final_modified_adj_chk.copy().tolil()

                print(len(perterbation_list))
                list_of_lists = perterbation_list
                #list_of_lists = list_of_lists.tolist()
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                print(list_to_remove)
                filtered_list_of_lists= [x for x in list_of_lists if x != list_to_remove]
                #filtered_list_of_lists = [lst for lst in list_of_lists if lst not in list_to_remove]
                perterbation_list = filtered_list_of_lists #np.array(filtered_list_of_lists)
                #perterbation_list=perterbation_list.tolist()
                print(len(perterbation_list))

                if len(perterbation_list) == 0:
                    break
                #print(len(filtered_list_of_lists))

                print(type(final_modified_adj_chk))
                print(final_modified_adj_chk)


            #acc_gcn = test1(final_modified_adj_chk, features, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            #print('class : %s' % acc_gcn)
            #if acc_gcn == 0:
                #cnt_gcn += 1

            acc_gat = test_GAT(final_modified_adj_chk, data, target_node) #single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' %acc_gat)
            if acc_gat == 0:
              cnt_gat += 1


        print('========= TESTING Same Orbit Connection =========')

        #print('Missclassification rate for GCN : %s' % (cnt_gcn / num))
        print('Missclassification rate for GAT : %s' % (cnt_gat / num))

        miss[0][gp]= cnt_gat / num
        #miss[1][gp] = cnt_gat / num



######################## Netattacks  #####################

        cnt_gcn = 0
        cnt_gat = 0
        degrees = adj.sum(0).A1

        for target_node in tqdm(node_list):
            #n_perturbations = 1 #math.ceil(degrees[target_node] * edg_pub_rate)  # int(degrees[target_node] * edg_pub_rate)
            #n_perturbations = math.ceil(degrees[target_node])

            #if budget >= 5:
            #n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            #else:

            n_perturbations= budget  #int(degrees[target_node])

            model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device)
            model = model.to(device)
            model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
            modified_adj = model.modified_adj

            print(type(modified_adj))
            print(modified_adj)

            # bst_edge[]= model.best_edge_list
            # bst_edge[target_node] = model.best_edge_list
            # modified_features = model.modified_features
            #acc_gcn = test1(modified_adj, features, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            #print('class : %s' % acc_gcn)
            #if acc_gcn == 0:
               # cnt_gcn += 1

            acc_gat = test_GAT(modified_adj, data, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' % acc_gat)
            if acc_gat == 0:
                cnt_gat += 1
            #acc = test_GAT(modified_adj, data, target_node)  ######### single_test(modified_adj, features, target_node, gcn=target_gcn)
            #bst_edge[target_node] = [model.best_edge_list, acc]
            #print('class : %s' % acc_gat)
            # Stop the timer
            # end_time = time.time()

            # Calculate the elapsed time
            # elapsed_time = end_time - start_time
            # Print the elapsed time
            # print(f"Elapsed time: {elapsed_time} seconds")

            #if acc == 0:
                #cnt1 += 1

        print('=== Testing Nettacks main (add/del/feature/all) ===')

        print('Miss-classification rate GCN: %s' % (cnt_gat / num))
        #print('Miss-classification rate GAT: %s' % (cnt_gat / num))

        miss[2][gp] = cnt_gat / num
        #miss[3][gp] = cnt_gat / num

############## RANDOM ATTACKS  #############


        cnt_gcn = 0
        cnt_gat = 0
        degrees = adj.sum(0).A1

        for target_node in tqdm(node_list):
            # n_perturbations = 1 #math.ceil(degrees[target_node] * edg_pub_rate)  # int(degrees[target_node] * edg_pub_rate)
            # n_perturbations = math.ceil(degrees[target_node])

            # if budget >= 5:
            # n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            # else:

            n_perturbations = budget ##int(degrees[target_node])
            model = RND()
            model = model.to(device)

            # Attack
            model.attack(adj, labels, idx_train, target_node, n_perturbations)
            modified_adj = model.modified_adj

            #model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
            #model = model.to(device)
            #model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
            #modified_adj = model.modified_adj

            print(type(modified_adj))
            print(modified_adj)

            # bst_edge[]= model.best_edge_list
            # bst_edge[target_node] = model.best_edge_list
            # modified_features = model.modified_features
            acc_gcn = test_GAT(modified_adj, data, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' % acc_gcn)
            if acc_gcn == 0:
                cnt_gat += 1

            #acc_gat = test_GAT(modified_adj, data, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
           #print('class : %s' % acc_gat)
           #if acc_gat == 0:
               #cnt_gat += 1
            # acc = test_GAT(modified_adj, data, target_node)  ######### single_test(modified_adj, features, target_node, gcn=target_gcn)
            # bst_edge[target_node] = [model.best_edge_list, acc]
            #rint('class : %s' % acc_gat)
            # Stop the timer
            # end_time = time.time()

            # Calculate the elapsed time
            # elapsed_time = end_time - start_time
            # Print the elapsed time
            # print(f"Elapsed time: {elapsed_time} seconds")

            # if acc == 0:
            # cnt1 += 1

        print('=== Testing Random ===')

        print('Miss-classification rate GCN: %s' % (cnt_gat / num))
        #print('Miss-classification rate GAT: %s' % (cnt_gat / num))

        miss[3][gp] = cnt_gat / num
        # miss[3][gp] = cnt_gat / num

##################################### 1518 Not same orbit connection ##############################

        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0

        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(
                adj).tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = utils.to_scipy(features).tolil()
            modified_features = utils.to_scipy(features).tolil()
        else:
            ori_adj = adj.tolil()
            modified_adj = adj.tolil()
            lst_modified_adj = adj.tolil()  ########## for comparison only
            work_modified_adj = adj.tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = features.tolil()
            modified_features = features.tolil()

        for target_node in tqdm(node_list):
            ############ Define the column name and the target value ########
            column_name = 'two_Orbit_type'
            Target_node = target_node
            #Target_node_orbit = [1518] #df_2d.iloc[Target_node, 1]
            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == '1518'].tolist()

            print("matching_indices")
            print(matching_indices)
            print(len(matching_indices))

            ################## Get neighbors of a specific node  ##################

            #numbers = list(range(0, len(df_2d))) #list(G.neighbors(target_node))
            #print("numbers")
            #print(numbers)
            #print(len(numbers))

            #matching_indices_revised = [x for x in numbers if x not in matching_indices]

            perterbation_list = [[Target_node, value] for value in matching_indices]

            # print(perterbation_list)

            # potential_edges = np.column_stack((np.tile(target_node, adj.shape[0] - 1), np.setdiff1d(np.arange(adj.shape[0]), target_node)))
            # perterbation_list = potential_edges.astype("int32")
            # perterbation_list =perterbation_list.tolist()

            print(perterbation_list)
            print(len(perterbation_list))

            # nofnode_modification_lst = [1]
            # nofnode_modification = nofnode_modification_lst[0]

            ########################### Using edge one by one  ##############################
            degrees = adj.sum(0).A1
            print(degrees[target_node])
            # if budget >= 5:
            # n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            # else:
            n_perturbations = budget  # int(degrees[target_node])

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                print(edge_with_score)

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])  # reverse=True)

                print(sorted_list)

                print(len(sorted_list))

                print(sorted_list[0])

                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[
                    sorted_list[0][0], sorted_list[0][1]]

                lst_modified_adj = final_modified_adj_chk.copy().tolil()

                print(len(perterbation_list))
                list_of_lists = perterbation_list
                # list_of_lists = list_of_lists.tolist()
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                print(list_to_remove)
                filtered_list_of_lists = [x for x in list_of_lists if x != list_to_remove]
                # filtered_list_of_lists = [lst for lst in list_of_lists if lst not in list_to_remove]
                perterbation_list = filtered_list_of_lists  # np.array(filtered_list_of_lists)
                # perterbation_list=perterbation_list.tolist()
                print(len(perterbation_list))

                if len(perterbation_list) == 0:
                    break
                # print(len(filtered_list_of_lists))

                print(type(final_modified_adj_chk))
                print(final_modified_adj_chk)

            #acc_gcn = test1(final_modified_adj_chk, features,
             #               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            #rint('class : %s' % acc_gcn)
            #if acc_gcn == 0:
             #   cnt_gcn += 1

            acc_gat = test_GAT(final_modified_adj_chk, data, target_node) #single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' %acc_gat)
            if acc_gat == 0:
              cnt_gat += 1

        print('========= TESTING with 1518 nodes =========')

        print('Missclassification rate for GCN : %s' % (cnt_gat / num))
        ### print('Missclassification rate for GAT : %s' % (cnt_gat / num))

        miss[4][gp] = cnt_gat / num
        # miss[1][gp] = cnt_gat / num
######################################### 1819  ###############################

        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0

        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(
                adj).tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = utils.to_scipy(features).tolil()
            modified_features = utils.to_scipy(features).tolil()
        else:
            ori_adj = adj.tolil()
            modified_adj = adj.tolil()
            lst_modified_adj = adj.tolil()  ########## for comparison only
            work_modified_adj = adj.tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = features.tolil()
            modified_features = features.tolil()

        for target_node in tqdm(node_list):
            ############ Define the column name and the target value ########
            column_name = 'two_Orbit_type'
            Target_node = target_node
            # Target_node_orbit = [1518] #df_2d.iloc[Target_node, 1]
            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == '1819'].tolist()

            print("matching_indices")
            print(matching_indices)
            print(len(matching_indices))

            ################## Get neighbors of a specific node  ##################

            # numbers = list(range(0, len(df_2d))) #list(G.neighbors(target_node))
            # print("numbers")
            # print(numbers)
            # print(len(numbers))

            # matching_indices_revised = [x for x in numbers if x not in matching_indices]

            perterbation_list = [[Target_node, value] for value in matching_indices]

            # print(perterbation_list)

            # potential_edges = np.column_stack((np.tile(target_node, adj.shape[0] - 1), np.setdiff1d(np.arange(adj.shape[0]), target_node)))
            # perterbation_list = potential_edges.astype("int32")
            # perterbation_list =perterbation_list.tolist()

            print(perterbation_list)
            print(len(perterbation_list))

            # nofnode_modification_lst = [1]
            # nofnode_modification = nofnode_modification_lst[0]

            ########################### Using edge one by one  ##############################
            degrees = adj.sum(0).A1
            print(degrees[target_node])
            # if budget >= 5:
            # n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            # else:
            n_perturbations = budget  # int(degrees[target_node])

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                print(edge_with_score)

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])  # reverse=True)

                print(sorted_list)

                print(len(sorted_list))

                print(sorted_list[0])

                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[
                    sorted_list[0][0], sorted_list[0][1]]

                lst_modified_adj = final_modified_adj_chk.copy().tolil()

                print(len(perterbation_list))
                list_of_lists = perterbation_list
                # list_of_lists = list_of_lists.tolist()
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                print(list_to_remove)
                filtered_list_of_lists = [x for x in list_of_lists if x != list_to_remove]
                # filtered_list_of_lists = [lst for lst in list_of_lists if lst not in list_to_remove]
                perterbation_list = filtered_list_of_lists  # np.array(filtered_list_of_lists)
                # perterbation_list=perterbation_list.tolist()
                print(len(perterbation_list))

                if len(perterbation_list) == 0:
                    break
                # print(len(filtered_list_of_lists))

                print(type(final_modified_adj_chk))
                print(final_modified_adj_chk)

            #acc_gcn = test1(final_modified_adj_chk, features,
             #               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            #print('class : %s' % acc_gcn)
            #if acc_gcn == 0:
             #   cnt_gcn += 1

            acc_gat = test_GAT(final_modified_adj_chk, data, target_node) #single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' %acc_gat)
            if acc_gat == 0:
              cnt_gat += 1

        print('========= TESTING with 1819 nodes =========')

        print('Missclassification rate for GCN : %s' % (cnt_gat / num))
        ### print('Missclassification rate for GAT : %s' % (cnt_gat / num))

        miss[5][gp] = cnt_gat / num
        # miss[1][gp] = cnt_gat / num

        ###############################  1519 ############################

        budget = edgemod[gp]
        cnt_gcn = 0
        cnt_gat = 0

        if type(adj) is torch.Tensor:
            ori_adj = utils.to_scipy(adj).tolil()
            lst_modified_adj = utils.to_scipy(adj).tolil()
            comp_modified_adj = utils.to_scipy(adj).tolil()  ########## for comparison only
            work_modified_adj = utils.to_scipy(
                adj).tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = utils.to_scipy(features).tolil()
            modified_features = utils.to_scipy(features).tolil()
        else:
            ori_adj = adj.tolil()
            modified_adj = adj.tolil()
            lst_modified_adj = adj.tolil()  ########## for comparison only
            work_modified_adj = adj.tolil()  ######## for working menas modify it by adding/removing edges
            ori_features = features.tolil()
            modified_features = features.tolil()

        for target_node in tqdm(node_list):
            ############ Define the column name and the target value ########
            column_name = 'two_Orbit_type'
            Target_node = target_node
            # Target_node_orbit = [1518] #df_2d.iloc[Target_node, 1]
            # Find the indices of rows where the column has the target value
            matching_indices = df_2d.index[df_2d[column_name] == '1519'].tolist()

            print("matching_indices")
            print(matching_indices)
            print(len(matching_indices))

            ################## Get neighbors of a specific node  ##################

            # numbers = list(range(0, len(df_2d))) #list(G.neighbors(target_node))
            # print("numbers")
            # print(numbers)
            # print(len(numbers))

            # matching_indices_revised = [x for x in numbers if x not in matching_indices]

            perterbation_list = [[Target_node, value] for value in matching_indices]

            # print(perterbation_list)

            # potential_edges = np.column_stack((np.tile(target_node, adj.shape[0] - 1), np.setdiff1d(np.arange(adj.shape[0]), target_node)))
            # perterbation_list = potential_edges.astype("int32")
            # perterbation_list =perterbation_list.tolist()

            print(perterbation_list)
            print(len(perterbation_list))

            # nofnode_modification_lst = [1]
            # nofnode_modification = nofnode_modification_lst[0]

            ########################### Using edge one by one  ##############################
            degrees = adj.sum(0).A1
            print(degrees[target_node])
            # if budget >= 5:
            # n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            # else:
            n_perturbations = budget  # int(degrees[target_node])

            for ep in range(n_perturbations):
                modified_adj_chk = lst_modified_adj.copy().tolil()
                ########### Getting score for each edge Addition/Deletion ##########

                edge_with_score = []

                ###### for one addition/deletion

                for i in perterbation_list:
                    modified_adj_chk_work = modified_adj_chk.copy().tolil()  ########## this will work as reset of the edge list
                    # print(modified_adj_chk_work[i[0], i[1]])

                    modified_adj_chk_work[i[0], i[1]] = 1 - modified_adj_chk_work[i[0], i[1]]

                    logits_modified = compute_logits(modified_adj_chk_work, modified_features, surrogate, target_node)
                    label_u = labels[target_node]
                    label_target_onehot = np.eye(int(nclass))[labels[target_node]]
                    best_wrong_class = (logits_modified - 1000 * label_target_onehot).argmax()
                    surrogate_losses_modified = logits_modified[labels[target_node]] - logits_modified[best_wrong_class]
                    edge_with_score.append([i[0], i[1], surrogate_losses_modified])

                print(edge_with_score)

                sorted_list = sorted(edge_with_score, key=lambda x: x[2])  # reverse=True)

                print(sorted_list)

                print(len(sorted_list))

                print(sorted_list[0])

                ################### final graph edge addition for testing ################

                final_modified_adj_chk = lst_modified_adj.copy().tolil()
                final_modified_adj_chk[sorted_list[0][0], sorted_list[0][1]] = 1 - final_modified_adj_chk[
                    sorted_list[0][0], sorted_list[0][1]]

                lst_modified_adj = final_modified_adj_chk.copy().tolil()

                print(len(perterbation_list))
                list_of_lists = perterbation_list
                # list_of_lists = list_of_lists.tolist()
                list_to_remove = [sorted_list[0][0], sorted_list[0][1]]
                print(list_to_remove)
                filtered_list_of_lists = [x for x in list_of_lists if x != list_to_remove]
                # filtered_list_of_lists = [lst for lst in list_of_lists if lst not in list_to_remove]
                perterbation_list = filtered_list_of_lists  # np.array(filtered_list_of_lists)
                # perterbation_list=perterbation_list.tolist()
                print(len(perterbation_list))

                if len(perterbation_list) == 0:
                    break
                # print(len(filtered_list_of_lists))

                print(type(final_modified_adj_chk))
                print(final_modified_adj_chk)

            #acc_gcn = test1(final_modified_adj_chk, features,
             #               target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            #print('class : %s' % acc_gcn)
            #if acc_gcn == 0:
             #   cnt_gcn += 1

            acc_gat = test_GAT(final_modified_adj_chk, data, target_node) #single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' %acc_gat)
            if acc_gat == 0:
              cnt_gat += 1

        print('========= TESTING with 1519 nodes =========')

        print('Missclassification rate for GCN : %s' % (cnt_gat / num))
        ### print('Missclassification rate for GAT : %s' % (cnt_gat / num))

        miss[6][gp] = cnt_gat / num
        # miss[1][gp] = cnt_gat / num

        ################ FGA #################

        cnt_gcn = 0
        cnt_gat = 0
        degrees = adj.sum(0).A1

        for target_node in tqdm(node_list):
            # n_perturbations = 1 #math.ceil(degrees[target_node] * edg_pub_rate)  # int(degrees[target_node] * edg_pub_rate)
            # n_perturbations = math.ceil(degrees[target_node])

            # if budget >= 5:
            # n_perturbations = math.ceil(degrees[target_node]) #* nofnode_modification)  # nofnode_modification #int(degrees[target_node])
            # else:

            n_perturbations = budget   ### int(degrees[target_node])

            model = FGA(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device='cpu').to('cpu')
            # Attack

            model.attack(features, adj, labels, idx_train, target_node, n_perturbations)
            modified_adj = model.modified_adj

            #model = RND()
            #model = model.to(device)

            # Attack
            #model.attack(adj, labels, idx_train, target_node, n_perturbations)
            #modified_adj = model.modified_adj

            # model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
            # model = model.to(device)
            # model.attack(features, adj, labels, target_node, n_perturbations, verbose=False)
            # modified_adj = model.modified_adj

            print(type(modified_adj))
            print(modified_adj)

            # bst_edge[]= model.best_edge_list
            # bst_edge[target_node] = model.best_edge_list
            # modified_features = model.modified_features
            acc_gcn = test_GAT(modified_adj, data,target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
            print('class : %s' % acc_gcn)
            if acc_gcn == 0:
                cnt_gat += 1

            # acc_gat = test_GAT(modified_adj, data, target_node)  # single_test(modified_adj, features, target_node, gcn=target_gcn)
        # print('class : %s' % acc_gat)
        # if acc_gat == 0:
        # cnt_gat += 1
        # acc = test_GAT(modified_adj, data, target_node)  ######### single_test(modified_adj, features, target_node, gcn=target_gcn)
        # bst_edge[target_node] = [model.best_edge_list, acc]
        # rint('class : %s' % acc_gat)
        # Stop the timer
        # end_time = time.time()

        # Calculate the elapsed time
        # elapsed_time = end_time - start_time
        # Print the elapsed time
        # print(f"Elapsed time: {elapsed_time} seconds")

        # if acc == 0:
        # cnt1 += 1

        print('=== Testing FGA ===')

        print('Miss-classification rate GCN: %s' % (cnt_gat / num))
        # print('Miss-classification rate GAT: %s' % (cnt_gat / num))

        miss[8][gp] = cnt_gat / num

    # miss[3][gp] = cnt_gat / num
    #print(miss)



print(miss)
df = pd.DataFrame(miss)
    # Save the DataFrame as a CSV file
    #df.to_csv('same_orbit_miss.csv', index=False)
df.to_csv(filename, index=False)