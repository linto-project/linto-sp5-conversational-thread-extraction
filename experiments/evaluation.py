import os
import csv
import networkx as nx
import pandas as pd
from math import log
import typing as ty
import glob
from fileinput import filename

# This is from https://github.com/LoicGrobol/scorch
from scores import ceaf_e, muc, b_cubed, ceaf 


# Established empirically on dev, train
MAX_DEGREE = 15
MAX_DISTANCE = 13

###############################################################################
## Requires a file whose lines contain links of the form
#  file_id:m1 m2 -
# Those are the files used by Kummerfald et al 2019 scripts  
def extractGraphFromTextPairs(file):
    
    graphs = dict()
    
    with open(file) as gold_file :
        links = gold_file.readlines()
        links = [line.strip() for line in links]
    
    
    for link in links :
        file_id = link.split(':')[0]
        e1 = int(link.split(':')[1].split(' ')[0])
        e2 = int(link.split(':')[1].split(' ')[1])
        
        if file_id not in graphs : 
            graph = nx.DiGraph(file_id=file_id)
            if e1 not in graph:
                graph.add_node(e1, label=str(e1))
            if e2 not in graph:
                graph.add_node(e2, label=str(e2))
            graph.add_edge(e1, e2)
            
            graphs[file_id] = graph
        else :
            if e1 not in graphs[file_id]:
                graphs[file_id].add_node(e1, label=str(e1))
            if e2 not in graphs[file_id]:
                graphs[file_id].add_node(e2, label=str(e2))
            graphs[file_id].add_edge(e1, e2)
            
    # Add self loops
    for file, graph in graphs.items() :
        add_self_loops(graph)
            
    return graphs
    
    
###############################################################################
# Requires a dataframe with at least the following rows:
# - file_id // the id of the file
# - source // the source message
# - target // the target message
# - label // whether they are linked or not according to the gold annotations
# - pred // whether they are linked or not according to the prediction model
# 
# Returns three graphs:
# - one representing the gold annotations 
# - one representing the predictions
# -one representing predictions after decoding (cf satisfiesConstraints)
#
# In case that the input data contain links with nodes prior to 1000 we do not 
# keep them
def extractGraphs(data):
    
    gold_graphs = dict()
    pred_graphs = dict()
    pred_graphs_constraints = dict()
    
    for index, row in data.iterrows():
        e1 = int(row['source'])
        e2 = int(row['target'])
        if row['file_id'] not in pred_graphs:
            gold = nx.DiGraph(file_id=row['file_id'])
            pred = nx.DiGraph(file_id=row['file_id'])
            pred_constr = nx.DiGraph(file_id=row['file_id'])
            # We add the nodes
            if e1 not in gold:
                gold.add_node(e1, label=str(e1))
            if e2 not in gold:
                gold.add_node(e2, label=str(e2)) # predicted labels, no constraints
            if e1 not in pred:
                pred.add_node(e1, label=str(e1))
            if e2 not in pred:
                pred.add_node(e2, label=str(e2)) # predicted labels, with constraints
            if e1 not in pred_constr:
                pred_constr.add_node(e1, label=str(e1))
            if e2 not in pred_constr:
                pred_constr.add_node(e2, label=str(e2)) # scores with constraints
            # add edges
            if int(row['label']) == 1:
                gold.add_edge(e1, e2) # predicted labels, no constraints
            if int(row['pred']) == 1:
                pred.add_edge(e1, e2) # predicted labels, with constraints
            if int(row['pred']) == 1 and satisfiesConstraints(pred_constr, e1, e2):
                pred_constr.add_edge(e1, e2) # scores with constraints; note that our dataframe is sorted (desceding) according to the score for class 1

            
            # Remove nodes < 1000
            if e1 < 1000 : 
                gold.remove_node(e1)
                pred.remove_node(e1)
                pred_constr.remove_node(e1)
            if e2 < 1000 : 
                gold.remove_node(e2)
                pred.remove_node(e2)
                pred_constr.remove_node(e2)
            
            gold_graphs[row['file_id']] = gold
            pred_graphs[row['file_id']] = pred
            pred_graphs_constraints[row['file_id']] = pred_constr
        else:
            # We add the nodes
            if e1 not in gold_graphs[row['file_id']]:
                gold_graphs[row['file_id']].add_node(e1, label=str(e1))
            if e2 not in gold_graphs[row['file_id']]:
                gold_graphs[row['file_id']].add_node(e2, label=str(e2)) # predicted labels, no constraints
            if e1 not in pred_graphs[row['file_id']]:
                pred_graphs[row['file_id']].add_node(e1, label=str(e1))
            if e2 not in pred_graphs[row['file_id']]:
                pred_graphs[row['file_id']].add_node(e2, label=str(e2)) # predicted labels, with constraints
            if e1 not in pred_graphs_constraints[row['file_id']]:
                pred_graphs_constraints[row['file_id']].add_node(e1, label=str(e1))
            if e2 not in pred_graphs_constraints[row['file_id']]:
                pred_graphs_constraints[row['file_id']].add_node(e2, label=str(e2)) # scores with constraints
            # gold
            if int(row['label']) == 1:
                gold_graphs[row['file_id']].add_edge(e1, e2) # predicted labels, no constraints
            if int(row['pred']) == 1:
                pred_graphs[row['file_id']].add_edge(e1, e2) # predicted labels, with constraints
            if int(row['pred']) == 1 and satisfiesConstraints(pred_graphs_constraints[row['file_id']], e1, e2):
                pred_graphs_constraints[row['file_id']].add_edge(e1, e2) # scores with constraints; note that our dataframe is sorted (desceding) according to the score for class 1
            
            # Remove nodes < 1000
            if e1 < 1000 : 
                gold_graphs[row['file_id']].remove_node(e1)
                pred_graphs[row['file_id']].remove_node(e1)
                pred_graphs_constraints[row['file_id']].remove_node(e1)
            if e2 < 1000 : 
                gold_graphs[row['file_id']].remove_node(e2)
                pred_graphs[row['file_id']].remove_node(e2)
                pred_graphs_constraints[row['file_id']].remove_node(e2)


    # Add self loops
    for file, graph in gold_graphs.items() :
        add_self_loops(graph)
    for file, graph in pred_graphs.items() :
        add_self_loops(graph)
    for file, graph in pred_graphs_constraints.items() :
        add_self_loops(graph)
    

    return gold_graphs, pred_graphs, pred_graphs_constraints              


###########################################################################

def add_self_loops(graph):
    
    # firstly add self loops to all isolated nodes
    for cc in sorted(nx.weakly_connected_components(graph), key=len, reverse=True) :
        if len(cc) == 1 :
            node = next(iter(cc)) 
            graph.add_edge(node, node)
            
    # Then add a self loop to all nodes that have an in degree of 0
    for node in graph.nodes :
        if graph.in_degree(node) == 0 :
            graph.add_edge(node, node)


###########################################################################

def b_cubed_eval(gold_graphs, pred_graphs):
    
    P, R = 0.0, 0.0
    
    for graph_file, gold in gold_graphs.items():
        predCCs = sorted(nx.weakly_connected_components(pred_graphs[graph_file]), key=len, reverse=True)
        goldCCs = sorted(nx.weakly_connected_components(gold), key=len, reverse=True)
        
        r, p, f = b_cubed(goldCCs, predCCs)
        
        P += p
        R += r 
    
    P = P / len(gold_graphs.items())
    R = R / len(gold_graphs.items())
    
    return (round (P, 4) * 100, 
            round (R, 4) * 100, 
            round ((2 * P * R / (P + R)), 4) * 100)  

###########################################################################



def phi4(gold_graphs, pred_graphs):
    
    P, R = 0.0, 0.0
    
    for graph_file, gold in gold_graphs.items():
        predCCs = sorted(nx.weakly_connected_components(pred_graphs[graph_file]), key=len, reverse=True)
        goldCCs = sorted(nx.weakly_connected_components(gold), key=len, reverse=True)
        
        r, p, f = ceaf_e(goldCCs, predCCs)
        
        P += p
        R += r 
        
    
    P = P / len(gold_graphs.items())
    R = R / len(gold_graphs.items())
    
    return (round (P, 4) * 100, 
            round (R, 4) * 100, 
            round ((2 * P * R / (P + R)), 4) * 100) 
    

###########################################################################

def muc_eval(gold_graphs, pred_graphs):
    P, R = 0.0, 0.0
    
    for graph_file, gold in gold_graphs.items():
        predCCs = sorted(nx.weakly_connected_components(pred_graphs[graph_file]), key=len, reverse=True)
        goldCCs = sorted(nx.weakly_connected_components(gold), key=len, reverse=True)
        
        r, p, f = muc(goldCCs, predCCs)
        
        P += p
        R += r 
    
    P = P / len(gold_graphs.items())
    R = R / len(gold_graphs.items())
    
    return (round (P, 4) * 100, 
            round (R, 4) * 100, 
            round ((2 * P * R / (P + R)), 4) * 100)
    


###########################################################################


def purity(gold_graphs, pred_graphs):
    
    total_nodes = 0
    total_overlap = 0
    
    for graph_file, gold in gold_graphs.items():
        predCCs = sorted(nx.weakly_connected_components(pred_graphs[graph_file]), key=len, reverse=True)
        goldCCs = sorted(nx.weakly_connected_components(gold), key=len, reverse=True)
        
        for predCC in predCCs :
            # find the highsest overlap
            overlap = 0
            for goldCC in goldCCs :
                overlap_set = predCC & goldCC
                if len(overlap_set) >= overlap : overlap = len(overlap_set)
            total_overlap += overlap
        total_nodes += gold.number_of_nodes()
        
    return float(total_overlap) / float(total_nodes)

###########################################################################


def satisfiesConstraints(graph, e1, e2):
    
    # Nodes shouldn't be over the MAX_DEGREE
    for node, degree in graph.degree([e1, e2]) :
        if degree >= MAX_DEGREE :
            return False
        
    # No backward links
    if e1 > e2 :
        return False
    
    if e2 - e1 >= MAX_DISTANCE :
        return False
    
    return True


################################################################################
# Outputs files in both formats required by the scripts of Kummerfeld et al. 2019
def write_eval_files(graphs, prefix):
    suffix_clusters = ".clusters.txt"
    suffix_graphs = ".graphs.txt"
    
    outGraphs = open(prefix + suffix_graphs, 'w')
    outClusters = open(prefix + suffix_clusters, 'w')
    for f, g in graphs.items() :
        # only edges
        for s, t in g.edges :
            outGraphs.write(f + ":" + str(s) + " " + str(t) + " -\n")
        # the clusters
        CCs = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
        for cc in CCs :
            cluster = f +":"
            for node in cc :
                cluster += str(node) + " "
            outClusters.write(cluster.strip() + "\n")
    outGraphs.close()
    outClusters.close()
    

################################################################################

def exctractDataframe(tsv_loc, pred_loc, score_loc):
    data = pd.read_csv(tsv_loc, sep="\t")
    with open(pred_loc) as preds_file:
        preds = preds_file.readlines()
    preds = [line.strip() for line in preds]
# print(len(preds))
# print(data.shape)
# append prediction to pandas data frame
    data['pred'] = preds
    with open(score_loc) as scores_file:
        scores = scores_file.readlines()
    scores_0 = [float(line.strip().strip('[]').strip().split()[0]) for line in scores]
    scores_1 = [float(line.strip().strip('[]').strip().split()[1]) for line in scores]
    data['scores_0'] = scores_0
    data['scores_1'] = scores_1
# pd.set_option("display.max_rows", None, "display.max_columns", None)
# print(data.head())
# print(list(data.columns))
    data = data.sort_values(by=['scores_1'], ascending=False)
    return data
################################################################################




# # You can run this script in the following way: 
# 
# #################################
# # tsv files:
# tsv_loc = "/path/to/tsv/file.tsv"
# pred_loc = "/path/to/eval_preds.txt"
# score_loc = "/path/to/eval_scores.txt"
#  
# ###### Get the data into graphs 
# data = exctractDataframe(tsv_loc, pred_loc, score_loc) 
# gold_graphs, pred_graphs, pred_graphs_constraints = extractGraphs(data)
# #################################
# # 
# # Alternatively output files from the Kummerfeld predictions
# 
# # gold_loc = "/path/to/kummerfeld/gold.test.graphs.txt"
# # pred_loc = "/path/to/kummerfeld_output/file.out"
# #  
# # gold_graphs = extractGraphFromTextPairs(gold_loc)
# # pred_graphs = extractGraphFromTextPairs(pred_loc)
# 
# 
# 
# ### Save on the format required by the Kummerfeld et al 2019 scripts
# write_eval_files(gold_graphs, "gold")
# write_eval_files(pred_graphs, "pred")
# 
# print("Purity:")
# print(round(purity(gold_graphs, pred_graphs), 4)*100)
# print(round(purity(gold_graphs, pred_graphs_constraints), 4)*100)
#   
# print("MUC:")
# print(muc_eval(gold_graphs, pred_graphs))
# print(muc_eval(gold_graphs, pred_graphs_constraints))
#   
# print("phi4:")
# print(phi4(gold_graphs, pred_graphs))
# print(phi4(gold_graphs, pred_graphs_constraints))
#    
# print("b^3:")
# print(b_cubed_eval(gold_graphs, pred_graphs))
# print(b_cubed_eval(gold_graphs, pred_graphs_constraints))

