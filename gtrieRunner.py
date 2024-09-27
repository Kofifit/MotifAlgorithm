from util_functions import UtilFunctions
import subprocess
import pandas as pd

def runAnalysisNauty(n, network):
    filename = 'bin/networkNauty.txt'
    col_names = ['edge', 'Activation/Repression', 'delta']
    UtilFunctions.network2txt(filename, network, col_names)
    subprocess.run(["./gtrieScannerFolder/gtrieScanner", "-s", str(n), "-d", "-m", "esu", "-g", filename, "-o", 'bin/results.txt', "-oc", "bin/locations.txt"],
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)
    result = convertResult2df(n, network, 'bin/locations.txt')
    return result

def convertResult2df(n, network, filename_locations):
    edge_indices = {}
    location_list = {}
    motif_dict = {}
    nodes_dict = {}
    with open(filename_locations) as f:
        for line in f:
            line = line.strip().split(':')
            motif = line[0]
            location = line[1].split()
            if motif not in motif_dict.keys():
                motif_converted = convertGraphBit2Dict(motif, n)
                motif_dict[motif] = motif_converted
                nodes_dict[motif] = []
                edge_indices[motif] = []
                location_list[motif] = []
            location = [int(i) for i in location]
            nodes_dict[motif].append(location)
            edge_indices[motif].append([])
            location_list[motif].append([])

    for index_edge, edge in network.items():
        for motif in nodes_dict.keys():
            for index_graph, graph in enumerate(nodes_dict[motif]):
                if set(edge[0]).issubset(set(graph)) and len(set(edge[0])) > 1:
                    edge_indices[motif][index_graph].append(index_edge)
                    location_list[motif][index_graph].append(edge[0])

    motifs = []
    counters = []
    indices = []
    locations = []
    for motif in motif_dict.keys():
        motifs.append(motif_dict[motif])
        indices.append([tuple(sub) for sub in edge_indices[motif]])
        locations.append(location_list[motif])
        counters.append(len(edge_indices[motif]))

    df = pd.DataFrame()
    df['Motif'] = motifs
    df['Number of appearances in network'] = counters
    df['Edges indices'] = indices
    df['Location of appearances in network'] = locations
    return df

def convertGraphBit2Dict(graph, n):
    graph_dict = {}
    counter = 0
    node = 0
    for char in graph:
        if int(char) == 1:
            if node not in graph_dict.keys():
                graph_dict[node] = []
            graph_dict[node].append(counter)
        counter += 1
        if counter >= n:
            counter = 0
            node += 1
    return graph_dict

def convertResult2dfDraft(network, filename_results, filename_locations):

    getGraph = False
    motif = ''
    results = {}
    with open(filename_results) as f:
        for line in f:
            line = line.strip().split()
            if not getGraph:
                if len(line) > 2:
                    if line[1] == 'Org_Freq':
                        getGraph = True
            else:
                if line:
                    motif = line[0] + motif
                    if len(line) > 1:
                        results[motif] = [int(line[1]), []]
                        motif = ''

    with open(filename_locations) as f:
        print_motifs = []
        for line in f:
            line = line.strip().split(':')
            motif = line[0]
            location = line[1]
            print_motifs.append(motif)
            try:
                results[motif][1].append(location)
            except:
                print(motif, 'NOT FOUND')
        print(set(print_motifs))
        print(results.keys())

    for motif in results.keys():
        locations = results[motif][1]
        locations_formatted = []
        indices_list = []
        for loc in locations:
            current_location = []
            current_indices = []
            loc = loc.strip().split()
            nodes = [eval(i) for i in loc]
            for key in network.keys():
                edge = network[key]
                if set(edge).issubset(set(nodes)):
                    current_location.append(edge)
                    current_indices.append(key)
            locations_formatted.append(current_location)
            indices_list.append(tuple(sorted(current_indices)))

        results[motif][1] = locations_formatted
        results[motif].append(indices_list)

    return results












