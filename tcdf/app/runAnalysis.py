import TCDF
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pylab
import copy
import matplotlib.pyplot as plt
import os
import sys

# os.chdir(os.path.dirname(sys.argv[0])) #uncomment this line to run in VSCode

def check_positive(value):
    """Checks if argument is positive integer (larger than zero)."""
    ivalue = int(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

def check_zero_or_positive(value):
    """Checks if argument is positive integer (larger than or equal to zero)."""
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s should be positive" % value)
    return ivalue

class StoreDictKeyPair(argparse.Action):
    """Creates dictionary containing datasets as keys and ground truth files as values."""
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships."""
    gtdata = pd.read_csv(gtfile, header=None)

    readgt=dict()
    effects = gtdata[1]
    causes = gtdata[0]
    delays = gtdata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k]=[]
    for i in range(len(effects)):
        key=effects[i]
        value=causes[i]
        readgt[key].append(value)
        pairdelays[(key, value)]=delays[i]
        gtnrrelations+=1
    
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                #find extended delay
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    delay=0
                    for i in range(len(p)-1):
                        delay+=pairdelays[(p[i+1], p[i])]
                    extendedgtdelays[(effect, cause)].append(delay)

    return extendedgtdelays, readgt, extendedreadgt


def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
                
            else:
                next
           
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)


def runTCDF(datafile):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)

    allcauses = dict()
    alldelays = dict()
    allreallosses=dict()
    allscores=dict()

    columns = list(df_data)
    for c in columns:
        idx = df_data.columns.get_loc(c)
        causes, causeswithdelay, realloss, scores = TCDF.findcauses(c, cuda=cuda, epochs=nrepochs, 
        kernel_size=kernel_size, layers=levels, log_interval=loginterval, 
        lr=learningrate, optimizername=optimizername,
        seed=seed, dilation_c=dilation_c, significance=significance, file=datafile)

        allscores[idx]=scores
        allcauses[idx]=causes
        alldelays.update(causeswithdelay)
        allreallosses[idx]=realloss

    return allcauses, alldelays, allreallosses, allscores, columns
    
def runTCDF_single_feature(datafile, feature):
    """Loops through all variables in a dataset and return the discovered causes, time delays, losses, attention scores and variable names."""
    df_data = pd.read_csv(datafile)
    
    columns = list(df_data)

    idx = df_data.columns.get_loc(feature)
    causes, causeswithdelay, realloss, scores = TCDF.findcauses(feature, cuda=cuda, epochs=nrepochs, 
    kernel_size=kernel_size, layers=levels, log_interval=loginterval, 
    lr=learningrate, optimizername=optimizername,
    seed=seed, dilation_c=dilation_c, significance=significance, file=datafile)

    return causes, causeswithdelay, realloss, scores, columns

def plotgraph(stringdatafile,alldelays,columns):
    """Plots a temporal causal graph showing all discovered causal relationships annotated with the time delay between cause and effect."""
    G = nx.DiGraph()
    for c in columns:
        G.add_node(c)
    for pair in alldelays:
        p1,p2 = pair
        nodepair = (columns[p2], columns[p1])

        G.add_edges_from([nodepair],weight=alldelays[pair])
    
    edge_labels=dict([((u,v,),d['weight'])
                    for u,v,d in G.edges(data=True)])
    
    pos=nx.circular_layout(G)
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
    nx.draw(G,pos, node_color = 'white', edge_color='black',node_size=1000,with_labels = True)
    ax = plt.gca()
    ax.collections[0].set_edgecolor("#000000") 

    pylab.show()

def main(datafiles, evaluation):
    if evaluation:
        totalF1direct = [] #contains F1-scores of all datasets
        totalF1 = [] #contains F1'-scores of all datasets

        receptivefield=1
        for l in range(0, levels):
            receptivefield+=(kernel_size-1) * dilation_c**(l)

    for datafile in datafiles.keys(): 
        stringdatafile = str(datafile)
        if '/' in stringdatafile:
            stringdatafile = str(datafile).rsplit('/', 1)[1]
        
        print("\n Dataset: ", stringdatafile)

        # run TCDF
        causes, delays, reallosses, scores, columns = runTCDF_single_feature(datafile, feature) #results of TCDF containing indices of causes and effects

        print(causes)
        print(delays)
        print(reallosses)
        print(scores)

        #write results to file
        with open(stringdatafile+"_results.txt", "w") as f:
            f.write("Discovered causes and time delays for feature "+feature+":\n")
            for pair in delays:
                p1,p2 = pair
                f.write(columns[p2]+" --> "+columns[p1]+", delay: "+str(delays[pair])+"\n")
                f.write("Attention scores for feature "+feature+":\n")
                for i, score in enumerate(scores):
                    f.write(columns[i]+": "+str(score)+"\n")
                f.write("\n\n")


parser = argparse.ArgumentParser(description='TCDF: Temporal Causal Discovery Framework')

parser.add_argument('--cuda', action="store_true", default=False, help='Use CUDA (GPU) (default: False)')
parser.add_argument('--epochs', type=check_positive, default=1000, help='Number of epochs (default: 1000)')
parser.add_argument('--kernel_size', type=check_positive, default=4, help='Size of kernel, i.e. window size. Maximum delay to be found is kernel size - 1. Recommended to be equal to dilation coeffient (default: 4)')
parser.add_argument('--hidden_layers', type=check_zero_or_positive, default=0, help='Number of hidden layers in the depthwise convolution (default: 0)') 
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate (default: 0.01)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'RMSprop'], help='Optimizer to use (default: Adam)')
parser.add_argument('--log_interval', type=check_positive, default=500, help='Epoch interval to report loss (default: 500)')
parser.add_argument('--seed', type=check_positive, default=1111, help='Random seed (default: 1111)')
parser.add_argument('--dilation_coefficient', type=check_positive, default=4, help='Dilation coefficient, recommended to be equal to kernel size (default: 4)')
parser.add_argument('--significance', type=float, default=0.8, help="Significance number stating when an increase in loss is significant enough to label a potential cause as true (validated) cause. See paper for more details (default: 0.8)")
parser.add_argument('--plot', action="store_true", default=False, help='Show causal graph (default: False)')
parser.add_argument('--feature', type=str, default='', help='Feature to use as basis for root cause analysis')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--ground_truth',action=StoreDictKeyPair, help='Provide dataset(s) and the ground truth(s) to evaluate the results of TCDF. Argument format: DataFile1=GroundtruthFile1,Key2=Value2,... with a key for each dataset containing multivariate time series (required file format: csv, a column with header for each time series) and a value for the corresponding ground truth (required file format: csv, no header, index of cause in first column, index of effect in second column, time delay between cause and effect in third column)')
group.add_argument('--data', nargs='+', help='(Path to) one or more datasets to analyse by TCDF containing multiple time series. Required file format: csv with a column (incl. header) for each time series')


args = parser.parse_args()

print("Arguments:", args)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, you should probably run with --cuda to speed up training.")
if args.kernel_size != args.dilation_coefficient:
    print("WARNING: The dilation coefficient is not equal to the kernel size. Multiple paths can lead to the same delays. Set kernel_size equal to dilation_c to have exaxtly one path for each delay.")

kernel_size = args.kernel_size
levels = args.hidden_layers+1
nrepochs = args.epochs
learningrate = args.learning_rate
optimizername = args.optimizer
dilation_c = args.dilation_coefficient
loginterval = args.log_interval
seed=args.seed
cuda=args.cuda
significance=args.significance
feature=args.feature

if args.ground_truth is not None:
    datafiles = args.ground_truth
    main(datafiles, evaluation=True)

else:
    datafiles = dict()
    for dataset in args.data:
        datafiles[dataset]=""
    main(datafiles, evaluation=False)
