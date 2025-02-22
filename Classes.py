from BruteForceAlgorithm import runAnalysis
from util_functions import UtilFunctions
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from gtrieRunner import runAnalysisNauty
import pandas as pd
import re
import itertools



class Network:
    """
    Represents a network of nodes and edges. The network can add nodes, edges,
    retrieve information about nodes and edges, and perform operations to manipulate
    its structure.
    """

    def __init__(self):
        """
        Initializes a Network object with empty nodes and edges dictionaries.
        """
        self.nodes = dict()  # Dictionary to store nodes
        self.edges = dict()  # Dictionary to store edges

    def getNodes(self):
        """
        Returns a list of all nodes in the network.
        """
        return self.nodes.values()

    def getEdges(self):
        """
        Returns a list of all edges in the network.
        """
        return self.edges.values()

    def addNode(self, node):
        """
        Adds a node to the network.

        Parameters:
            node (Node): Node object to be added to the network.
        """
        self.nodes.setdefault(node.getName(), node)

    def addEdge(self, edge):
        """
        Adds an edge to the network.

        Parameters:
            edge (Edge): Edge object to be added to the network.
        """
        self.edges.setdefault(edge.getIndex(), edge)

    def createNode(self, name):
        """
        Creates a new node if it doesn't exist already.

        Parameters:
            name (str): Name of the node to be created.

        Returns:
            Node: Newly created or existing Node object.
        """
        if self.isNodeIn(name):
            return self.nodes[name]
        else:
            return Node(name)

    def createEdge(self, index, edge):
        """
        Creates a new edge and its associated nodes if they don't exist already in the network.

        Parameters:
            index (int): Index of the edge.
            edge (list): List containing edge information: [node names, function, delta].
        """
        # Get node names from the edge
        nodeA_name = edge[0][0]
        nodeB_name = edge[0][1]
        # Create node object for each node
        nodeA = self.createNode(nodeA_name)
        nodeB = self.createNode(nodeB_name)
        # Create edge object using its nodes
        edgeObject = Edge(index, nodeA, nodeB, edge[1], edge[2])
        # Add the edge to the nodes objects
        nodeA.addEdge(edgeObject)
        nodeB.addEdge(edgeObject)
        # Add nodes to the network
        self.addNode(nodeA)
        self.addNode(nodeB)
        # Add edge to the network
        self.addEdge(edgeObject)

    def extractEdges(self):
        """
        Retrieve all edges from the nodes in the network and then add them to the edges of the network
        """
        for node in self.nodes.values():
            for edge in node.getEdges():
                self.addEdge(edge)

    def extractNodes(self):
        """
        Retrieve all nodes from the edges in the network and then add them to the nodes of the network
        """
        for edge in self.edges.values():
            for node in edge.getNodes():
                self.addNode(node)

    def isEdgeIn(self, edgeIndex):
        """
        Checks if an edge exists in the network.

        Parameters:
            edgeIndex (int): Index of the edge to check.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        if edgeIndex in self.edges.keys():
            return True
        return False

    def isNodeIn(self, nodeName):
        """
        Checks if a node exists in the network.

        Parameters:
            nodeName (str): Name of the node to check.

        Returns:
            bool: True if the node exists, False otherwise.
        """
        if nodeName in self.nodes.keys():
            return True
        return False

    def transferNetwork(self, network):
        """
        Transfers nodes and edges from one network to another.

        Parameters:
            network (Network): Another network object to transfer nodes and edges to.
        """
        for nodeName, node in self.nodes.items():
            network.addNode(node)
        for edgeIndex, edge in self.edges.items():
            network.addEdge(edge)



class Edge:
    """
    Represents an edge connecting two nodes in a network. An edge contains information
    about its source and target nodes, associated function, and delta value.
    """

    def __init__(self, index, sourceNode, targetNode, function, delta):
        """
        Initializes an Edge object with source node, target node, function, and delta.

        Parameters:
            index (int): Index of the edge.
            sourceNode (Node): Source node of the edge.
            targetNode (Node): Target node of the edge.
            function (str): Associated function of the edge.
            delta (int): Delta value of the edge.
        """
        self.index = index
        self.sourceNode = sourceNode
        self.targetNode = targetNode
        self.nodes = [sourceNode, targetNode] # List of nodes connected by the edge
        self.function = function
        self.delta = delta

    def getIndex(self):
        """
        Returns the index of the edge.
        """
        return self.index

    def getNodes(self):
        """
        Returns a list of nodes connected by the edge.
        """
        return self.nodes

    def isDelta(self):
        """
        Check if edge appears in the original network.
        If the edge does not appear in the original network, but appears in this network the delta would be 1.
        """
        if self.delta == 1 or self.delta == -1:
            return True
        return False


class Node:
    """
    Represents a node in a network. A node contains information about its name,
    neighboring nodes, and connected edges.
    """

    def __init__(self, name):
        """
        Initializes a Node object with a name and empty neighbors and edges dictionaries.

        Parameters:
            name (str): Name of the node.
        """

        self.name = name  # Name of the node
        self.neighbors = dict()  # Dictionary to store neighboring nodes
        self.edges = dict()  # Dictionary to store connected edges
        self.neighbor_edge_table = dict()  # Dictionary to store neighbor-edge pairs

    def getName(self):
        """
        Returns the name of the node.
        """
        return self.name

    def getNeighbors(self):
        """
        Returns a list of neighboring nodes.
        """
        return self.neighbors.values()

    def getEdges(self):
        """
        Returns a list of edges connected to the node.
        """
        return self.edges.values()

    def getEdgeByNeighbor(self, neighborName):
        """
        Returns an edge object by node name
        :param nodeName:
        :return:
        """
        return self.neighbor_edge_table[neighborName]

    def addEdge(self, edge):
        """
        Adds an edge to the node's edges dictionary and update neighboring nodes.

        Parameters:
            edge (Edge): Edge object to be added.
        """
        self.edges.setdefault(edge.getIndex(), edge)
        for node in edge.getNodes():
            if node.getName() != self.name:
                self.neighbors.setdefault(node.getName(), node)
                self.neighbor_edge_table.setdefault(node.getName(), edge)


class NetworkEssembler:
    """
    Assembles a new network object from a network that appear in a dictionary format.
    It creates a new network object and adds edges and nodes from the provided network.
    """

    def __init__(self, network):
        """
        Initializes a NetworkEssembler object and creates a network from a given network.

        Parameters:
            network (dict): Network in a dictionary format.
        """
        self.network = self.createNetwork(network)

    def getNetwork(self):
        """
        Returns the assembled network.
        """
        return self.network

    def createNetwork(self, network):
        """
        Creates a network from a given network in a dictionary format.

        Parameters:
            network (dict): Network in a dictionary format.

        Returns:
            Network: Assembled network object.
        """
        networkObject = Network()
        for index, edge in network.items():
            networkObject.createEdge(index, edge)
        return networkObject

class NetworkDisessembler:
    """
    Disassembles a network object into a network in a dictionary format.
    It creates a new network in a dictionary format and adds edges and nodes from the provided network.
    """

    def __init__(self, objectNetwork):
        """
         Initializes a NetworkDisessembler object and creates a network in a dictionary format from a given network
         object.

         Parameters:
             objectNetwork (Network): Network object to be disassembled.
         """
        self.network = self.createNetwork(objectNetwork)

    def getNetwork(self):
        """
        Returns the disassembled network.
        """
        return self.network

    def createNetwork(self, objectNetwork):
        """
        Creates a network in a dictionary format from a given network object.

        Parameters:
            objectNetwork (Network): Network object to be disassembled.

        Returns:
            dict: Disassembled network in a dictionary format.
        """
        network = dict()
        for edge in objectNetwork.getEdges():
            network[edge.getIndex()] = [[edge.sourceNode.getName(), edge.targetNode.getName()], edge.function, edge.delta]
        return network


class NetworkDeltaExtractor:
    """
    Extracts the delta network from the main network based on delta edges.
    The delta network contains the interactions that appear in the current network and not in the original network.
    In addition, it contains the areas around these new interactions based on the motif size.
    """

    def __init__(self, motif_size, network):
        """
        Initializes a NetworkDeltaExtractor object with a motif size and a network, and creates an empty delta network.

        Parameters:
            motif_size (int): Size of the motif for delta extraction.
            network (Network): Main network object.
        """
        self.motif_size = motif_size  # Motif size for delta extraction
        self.network = NetworkEssembler(network).getNetwork()  # Assemble the network
        self.deltaNetwork = Network()  # Initialize an empty delta network

    def extractDeltaNetwork(self):
        """
        Extracts the delta network from the main network based on delta edges.
        Use the 'getkDistanceNodeDown' function to explore and retrieve the nodes around the delta edges
        """
        edges = self.network.getEdges()
        # Iterate over each edge in the network
        for edge in edges:
            # Check if the edge is a delta edge -1 or +1
            if edge.isDelta():
                # Iterate over each node in the edge
                for node in edge.getNodes():
                    # Initialize a new Network object to store visited nodes and edges
                    visited = Network()
                    # Recursively explore nodes around the delta edge
                    currentDelta = self.getkDistanceNodeDown(node, self.motif_size - 2, visited)
                    # Transfer the explored network to the deltaNetwork
                    currentDelta.transferNetwork(self.deltaNetwork)

    def getDeltaNetwork(self):
        """
         Returns the extracted delta network.

         Returns:
             Network: Extracted delta network.
         """
        return self.deltaNetwork

    def getkDistanceNodeDown(self, node,  k, visited):
        """
        Recursively explores nodes down to a certain distance from the given node.
        This method is used to extract a sub-network (delta network) by exploring nodes
        down to a specified distance based on a given motif size.

        Parameters:
            node (Node): Current node to explore.
            k (int): Remaining distance to explore.
            visited (Network): Network object to store visited nodes and edges.

        Returns:
            Network: Network containing visited nodes and edges within the specified distance.
        """
        # Base Case
        if node is None or k < 0:
            return visited

        # Add current node to visited
        visited.addNode(node)

        # If there are remaining distance to explore
        if k > 0:
            # Add edges of the current node to the visited network
            for e in node.getEdges():
                if e.delta != -1:
                    visited.addEdge(e)
            # Recursively explore neighboring nodes
            for neighbor in node.getNeighbors():
                edge = node.getEdgeByNeighbor(neighbor.getName())
                # Only explore unvisited neighboring nodes
                if neighbor.getName() not in visited.nodes.keys() and edge.delta != -1:
                    self.getkDistanceNodeDown(neighbor, k - 1, visited)
        return visited


class DeltaNetworkMotifAnalyzer:
    """
    Analyzes delta network motifs and compares them with the original network.
    """

    def __init__(self, originNetwork, motif_size, analysis_type):
        """
        Initializes a DeltaNetworkMotifAnalyzer object with the original network and motif size.

        Parameters:
            originNetwork (Network): Original network to analyze.
            motif_size (int): Size of the motif for analysis.
        """
        self.originNetwork = originNetwork  # Original network for analysis
        self.motif_size = motif_size  # Motif size for analysis
        self.analysis_type = analysis_type
        self.originAnalysis = self.analyze(self.originNetwork)  # Perform analysis on the original network

    def analyze(self, network):
        """
        Performs motif analysis on the given network.

        Parameters:
            network (Network): Network to perform analysis on.

        Returns:
            DataFrame: DataFrame containing analysis results.
        """
        if self.analysis_type == 'BruteForce':
            df = runAnalysis(self.motif_size, network)
        elif self.analysis_type == 'Nauty':
            df = runAnalysisNauty(self.motif_size, network)
        return df

    def saveAnalysis(self, df, filename):
        """
        Saves analysis results to a CSV file.

        Parameters:
            df (DataFrame): DataFrame containing analysis results.
            filename (str): Name of the CSV file to save the analysis results.
        """

        df.to_csv(filename)

    def compare(self, network, delta_network, analysis):
        """
        Compares the analysis results between the original network and a modified network.
        This method identifies the common motifs for the modified network and the original,
        and adds these common motifs to the modified network analysis.

        Parameters:
            network (dict): Modified network for comparison.
            analysis (DataFrame): DataFrame containing analysis results for the modified network.

        Returns:
            DataFrame: DataFrame containing the compared analysis results.
        """
        originAnalysis_copy = self.originAnalysis.copy()

        check_submotif = {}
        # Iterate over each row (motif) in the original analysis
        for row_num, row in originAnalysis_copy.iterrows():
            origin_indices_keep = []
            motifs = row['Edges indices']
            check_submotif[row_num] = {}
            # Iterate over each motif's location in the analysis of the original network
            for index, motif in enumerate(motifs):
                keep = False
                check = False
                for edge in motif:
                    delta = network[edge][2]
                    # Check if the edge does not appear in the delta network
                    if edge not in delta_network:
                        # Keep motif in the original analysis
                        keep = True
                    else:
                        check = True

                    # Check if the edge does not appear in the modified network at all
                    if delta == -1:
                        # Remove motif from the original analysis
                        keep = False
                        break
                if keep:
                    origin_indices_keep.append(index)
                    if check:
                        check_submotif[row_num][index] = motif

            # Update the original analysis to remove edges that were not found in the modified network
            lst = list([edge for idx, edge in enumerate(motifs) if idx in origin_indices_keep])
            originAnalysis_copy.at[row_num, 'Edges indices'] = lst
            lst = list([loc for idx, loc in enumerate(row['Location of appearances in network']) if idx in origin_indices_keep])
            originAnalysis_copy.at[row_num, 'Location of appearances in network'] = lst
            originAnalysis_copy.at[row_num, 'Number of appearances in network'] = len(origin_indices_keep)


        ### Remove motifs with Trie Origin Analysis ###

        # Get all indices for each motif from the delta network
        delta_motif_indices = {}
        for row_num, motif in analysis.iterrows():
            delta_motif_indices[row_num] = motif['Edges indices']

        # Build a trie for motifs in the delta network
        delta_trie = MotifTrie()
        for motifs in delta_motif_indices.values():
            for motif in motifs:
                delta_trie.insert(motif)

        # Remove sub-motifs from origin analysis
        motifs_remove_origin = {}
        for row_num, motif_dict in check_submotif.items():
            motifs_remove_origin[row_num] = []
            for index, motif in motif_dict.items():
                if delta_trie.is_submotif(motif):
                    motifs_remove_origin[row_num].append(index)

        ### Remove motifs with Trie Delta Analysis ###

        # Build a trie for motifs in the origin network
        origin_trie = MotifTrie()
        for motifs in check_submotif.values():
            for motif in motifs.values():
                origin_trie.insert(motif)

        # Remove sub-motifs from delta analysis
        motifs_remove_delta = {}
        for row_num, motif_dict in delta_motif_indices.items():
            motifs_remove_delta[row_num] = []
            for index, motif in enumerate(motif_dict):
                if origin_trie.is_submotif(motif):
                    motifs_remove_delta[row_num].append(index)


        ## UPDATE LISTS OF MOTIFS ##

        # Remove sub-motifs from list for final subset analysis patch
        for row_num, indices in motifs_remove_origin.items():
            if row_num in check_submotif:
                for index in indices:
                    check_submotif[row_num].pop(index, None)
        for row_num, indices in motifs_remove_delta.items():
            if row_num in delta_motif_indices:
                for index in indices:
                    delta_motif_indices[row_num].pop(index, None)


        ### PATCH - Remove undetected submotifs with Set Origin Analysis###

        # Check for sub-motifs in the origin analysis
        motifs_remove = {}
        other_motifs = []
        for inner_dict in delta_motif_indices.values():
            other_motifs.extend(inner_dict)
        other_motifs_sets = [set(motif) for motif in other_motifs]
        sorted_other_motifs = sorted(other_motifs_sets, key=len, reverse=True)

        for row_num, location_dict in check_submotif.items():
            motifs_remove[row_num] = []
            for index, motif in location_dict.items():
                if self.is_submotif(set(motif), sorted_other_motifs):
                    motifs_remove[row_num].append(index)

        for row_num, row in originAnalysis_copy.iterrows():
            originAnalysis_copy.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(row['Edges indices']) if not idx in motifs_remove[row_num]]
            originAnalysis_copy.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if not idx in motifs_remove[row_num]]
            originAnalysis_copy.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(motifs_remove[row_num])

        ### PATCH - Remove undetected submotifs with Set Delta Analysis###

        # Check for sub-motifs in the current solution
        motifs_remove = {}
        other_motifs = []
        for inner_dict in check_submotif.values():
            other_motifs.extend(inner_dict.values())
        other_motifs_sets = [set(motif) for motif in other_motifs]
        sorted_other_motifs = sorted(other_motifs_sets, key=len, reverse=True)

        for row_num, locations in delta_motif_indices.items():
            motifs_remove[row_num] = []
            for index, motif in enumerate(locations):
                if self.is_submotif(set(motif), sorted_other_motifs):
                    motifs_remove[row_num].append(index)

        for row_num, row in analysis.iterrows():
            analysis.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(row['Edges indices']) if not idx in motifs_remove[row_num]]
            analysis.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx not in motifs_remove[row_num]]
            analysis.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(motifs_remove[row_num])

        # Merge the two solutions after removal of sub-motifs
        for row_num, row in originAnalysis_copy.iterrows():
            # If the motif still has appearances in the analysis of the modified network, add them to the copy original analysis
            if originAnalysis_copy['Number of appearances in network'].loc[row_num] > 0:
                i = analysis['Motif'][analysis['Motif'] == row['Motif']].index
                if i.empty:
                    analysis = analysis._append(originAnalysis_copy.loc[row_num], ignore_index=True)
                else:
                    i = i[0]
                    analysis.at[i, 'Edges indices'].extend(originAnalysis_copy.at[row_num, 'Edges indices'])
                    analysis.at[i, 'Location of appearances in network'].extend(
                        originAnalysis_copy.at[row_num, 'Location of appearances in network'])
                    analysis.at[i, 'Number of appearances in network'] += originAnalysis_copy.at[
                        row_num, 'Number of appearances in network']

        # # Build a trie for motifs in the delta network
        # origin_trie = MotifTrie()
        # for motifs in check_submotif.values():
        #     for motif in motifs.values():
        #         origin_trie.insert(motif)
        #
        # # Remove sub-motifs from originAnalysis_copy
        # motifs_remove = {}
        # for row_num, motif_dict in delta_motif_indices.items():
        #     motifs_remove[row_num] = []
        #     for index, motif in enumerate(motif_dict):
        #         if origin_trie.is_submotif(motif):
        #             motifs_remove[row_num].append(index)

        # for row_num, row in analysis.iterrows():
        #     analysis.at[row_num, 'Edges indices'] = [edge for idx, edge in enumerate(row['Edges indices']) if not idx in motifs_remove[row_num]]
        #     analysis.at[row_num, 'Location of appearances in network'] = [loc for idx, loc in enumerate(row['Location of appearances in network']) if idx not in motifs_remove[row_num]]
        #     analysis.at[row_num, 'Number of appearances in network'] = row['Number of appearances in network'] - len(motifs_remove[row_num])


        return analysis

    def is_submotif(self, subgraph_indices, all_indices):
        for graph in all_indices:
            if len(graph) > len(subgraph_indices):
                if subgraph_indices.issubset(graph):
                    return True
            else:
                return False
        return False


class MotifSearcher:

    def __init__(self, motifs_file, n):
        self.motifs = self.getMotifs(motifs_file)
        self.motif_size = n

    def getMotifs(self, motifs_file):
        motifs = []
        file = open(motifs_file, 'r')
        lines = file.readlines()

        for line in lines:
            current_motif = {}
            nodes_list = line.strip().split(';')
            for node in nodes_list:
                source = int(node.split(':')[0])
                target_genes = [int(s) for s in re.findall(r'\d+', node.split(':')[1])]
                current_motif[source] = target_genes
            motifs.append(current_motif)

        return np.array(motifs)

    def findMotifs(self, analysis):
        found = np.full(len(self.motifs), False)
        possCombination = list(itertools.permutations(range(self.motif_size)))
        for index, row in analysis.iterrows():
            current_motif = row['Motif']
            nodes_list = []
            for node in current_motif.keys():
                nodes_list.append(node)
                nodes_list.extend(current_motif[node])
            nodes_list = set(nodes_list)
            keep = False
            for combo in possCombination:
                if keep:
                    break
                combo_dict = UtilFunctions.get_label_dict(nodes_list, combo)
                buffer_subgraph = UtilFunctions.get_buffer_graph(combo_dict, current_motif)
                buffer_subgraph.pop('Matched')
                for motif_index, motif in enumerate(self.motifs):
                    if buffer_subgraph == motif:
                        found[motif_index] = True
                        keep = True
                        break
            if not keep:
                analysis.drop([index], inplace=True)

        print('The following motifs were searched for - ')
        print(self.motifs)
        print('Here are the motifs that were found in the network - ')
        print(self.motifs[found])
        print('Here are the motifs that were missing from the network - ')
        print(self.motifs[found == False])

        return analysis


class GraphVisualization:

    def __init__(self, solutions, analyses_lst):
        self.solutions = solutions
        self.analyses = analyses_lst

    def getGraphs(self, X):
        plt.rcParams.update({'font.size': 30})
        folder = 'Figures/'
        ## Draw combination graph for all solutions
        self.createCombinationGraph(self.solutions, folder + f"network{X}_Graph-Combined_solutions")
        for i, s in enumerate(self.solutions):
            self.createDeltaNetworkGraph(s, folder + f"network{X}_solution{i}_delta_graph")
            self.createRegularGraph(s, folder + f"network{X}_solution{i}_regular_graph")
            motifs_df = self.analyses[i]
            new_network = UtilFunctions.addMotifs2Network(s, motifs_df)
            for motif_index, m in motifs_df.iterrows():
                self.createMotifDeltaNetworkGraph(new_network,
                                                         folder + f"network{X}_solution{i}_motif{motif_index}_delta_motif_graph",
                                                         motif_index)
                self.createMotifNetworkGraph(new_network,
                                                    folder + f"network{X}_solution{i}_motif{motif_index}_regular_motif_graph",
                                                    motif_index)

    def createRegularGraph(self, network, name):
        network = UtilFunctions.Network2NetworkX(network)
        edges = network.edges()

        pos_dict = {}
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['pos'] = nx.spectral_layout(network)

        pos_dict['neato'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='neato')
        # pos_dict['spring']  = nx.spring_layout(network, k=0.1, iterations=100, seed=42)
        # pos_dict['sfdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='sfdp')
        # pos_dict['fdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='fdp')
        for pos_name, pos in pos_dict.items():
            node_size, fig_size = self.getSizes(network)
            fig, ax = plt.subplots(figsize=fig_size)
            # Draw nodes
            nx.draw_networkx_nodes(
                network,
                pos,
                node_size=node_size,
                node_color='lightblue',
                alpha=0.9,
            )

            # Draw edges
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 1],
                arrowstyle='->',
                width=2,
                arrowsize=30,
                alpha=0.8,
            )
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 2],
                arrowstyle='-[',
                width=2,
                arrowsize=10,
                alpha=0.8,
            )

            # Draw labels
            nx.draw_networkx_labels(network, pos, font_size=12)

            plt.suptitle(name)
            plt.savefig(name + '.png')
            plt.close()



    def createMotifNetworkGraph(self, network, name, motif_index):
        network = UtilFunctions.Network2NetworkX(network)
        edges = network.edges()
        colors = [[], []]
        for u, v in edges:
            if network[u][v]['function'] == 1:
                index = 0
            else:
                index = 1

            motifs = network[u][v]['motifs']
            color = 'black'
            if motif_index in motifs:
                color = 'red'
            colors[index].append(color)

        pos_dict = {}
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['pos'] = nx.spectral_layout(network)

        pos_dict['neato'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='neato')
        # pos_dict['spring']  = nx.spring_layout(network, k=0.1, iterations=100, seed=42)

        # pos_dict['sfdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='sfdp')
        # pos_dict['fdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='fdp')
        for pos_name, pos in pos_dict.items():
            node_size, fig_size = self.getSizes(network)
            plt.figure(figsize=fig_size)  # Adjust figure size
            # Draw nodes
            nx.draw_networkx_nodes(
                network,
                pos,
                node_size=node_size,
                node_color='lightblue',
                alpha=0.6
            )

            # Draw edges
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 1],
                edge_color=colors[0],
                width=2,
                arrowstyle='->',
                arrowsize=30,
                alpha=0.8
            )
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 2],
                edge_color=colors[1],
                width=2,
                arrowstyle='-[',
                arrowsize=10,
                alpha=0.8
            )

            # Draw labels
            nx.draw_networkx_labels(network, pos, font_size=12)

            plt.suptitle(name)
            plt.savefig(name + '.png')
            plt.close()


    def createMotifDeltaNetworkGraph(self, network, name, motif_index):
        network = UtilFunctions.Network2NetworkX(network)
        edges = network.edges()
        colors = [[], []]
        widths = [[], []]
        for u, v in edges:

            if network[u][v]['function'] == 1:
                index = 0
            else:
                index = 1

            motifs = network[u][v]['motifs']
            width = 0.5
            if motif_index in motifs:
                width = 3
            widths[index].append(width)

            delta = network[u][v]['delta']
            color = 'black'
            if delta == -1:
                color = 'red'
            elif delta == 1:
                color = 'darkgreen'
            colors[index].append(color)

        pos_dict = {}
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['spring']  = nx.spring_layout(network, k=0.1, iterations=100, seed=42)
        # pos_dict['pos'] = nx.spectral_layout(network)

        pos_dict['neato'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='neato')
        # pos_dict['sfdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='sfdp')
        # pos_dict['fdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='fdp')

        for pos_name, pos in pos_dict.items():
            node_size, fig_size = self.getSizes(network)
            plt.figure(figsize=fig_size)  # Adjust figure size
            # Draw nodes
            nx.draw_networkx_nodes(
                network,
                pos,
                node_size=node_size,
                node_color='lightblue',
                alpha=0.6
            )

            # Draw edges
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 1],
                width=widths[0],
                edge_color=colors[0],
                arrowstyle='->',
                arrowsize=30,
                alpha=0.8
            )
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 2],
                width=widths[1],
                edge_color=colors[1],
                arrowstyle='-[',
                arrowsize=10,
                alpha=0.8
            )


            # Draw labels
            nx.draw_networkx_labels(network, pos, font_size=12)

            plt.suptitle(name)
            plt.savefig(name + '.png')
            plt.close()

    def createCombinationGraph(self, solutions, name):
        merged_solutions = UtilFunctions.CombineSolutions(solutions)
        network = UtilFunctions.Network2NetworkX(merged_solutions)

        edges = network.edges()
        weights = [network[u][v]['weight'] for u, v in edges]

        # Position nodes using spring layout
        pos_dict = {}
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['pos'] = nx.spectral_layout(network)

        pos_dict['neato'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='neato')
        # pos_dict['sfdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='sfdp')

        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['spring']  = nx.spring_layout(network, k=0.1, iterations=100, seed=42)


        # Node size and edge weight scaling
        node_size = [1 + network.degree(n) for n in network.nodes()]
        max_weight = max(weights) if weights else 1
        scaled_weights = [w / max_weight * 3 for w in weights]

        # Visualization
        for pos_name, pos in pos_dict.items():
            node_size, fig_size = self.getSizes(network)
            plt.figure(figsize=fig_size)  # Adjust figure size
            # Draw nodes
            nx.draw_networkx_nodes(
                network,
                pos,
                node_size=node_size,
                node_color='lightblue',
                alpha=0.6
            )

            # Draw edges
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 1],
                width=scaled_weights,
                arrowstyle='->',
                arrowsize=30,
                alpha=0.8
            )
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 2],
                width=scaled_weights,
                arrowstyle='-[',
                arrowsize=10,
                alpha=0.8
            )


            # Draw labels
            nx.draw_networkx_labels(network, pos, font_size=12)

            plt.suptitle(name)
            plt.savefig(name + '.png')
            plt.close()

    def createDeltaNetworkGraph(self, deltaNetwork, name):
        network = UtilFunctions.Network2NetworkX(deltaNetwork)
        edges = network.edges()
        colors = [[], []]
        scaled_weights = [[], []]
        for u, v in edges:
            delta = network[u][v]['delta']
            if network[u][v]['function'] == 1:
                index = 0
            else:
                index = 1
            if delta == -1:
                color = 'red'
                weight = 3
            elif delta == 1:
                color = 'darkgreen'
                weight = 3
            else:
                color = 'black'
                weight = 0.5

            colors[index].append(color)
            scaled_weights[index].append(weight)
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)

        pos_dict = {}
        # pos_dict['pos'] = nx.kamada_kawai_layout(network)
        # pos_dict['pos'] = nx.spectral_layout(network)
        # pos_dict['spring']  = nx.spring_layout(network, k=0.1, iterations=100, seed=42)

        pos_dict['neato'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='neato')
        # pos_dict['sfdp'] = nx.drawing.nx_agraph.graphviz_layout(network, prog='sfdp')


        for pos_name, pos in pos_dict.items():
            node_size, fig_size = self.getSizes(network)
            plt.figure(figsize=fig_size)

            nx.draw_networkx_nodes(
                network,
                pos,
                node_size=node_size,
                node_color='lightblue',
                alpha=0.6
            )

            # Draw edges
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 1],
                arrowstyle='->',
                width = scaled_weights[0],
                edge_color=colors[0],
                arrowsize=30,
                alpha=0.8
            )
            nx.draw_networkx_edges(
                network,
                pos,
                edgelist=[(u, v) for u, v in edges if network[u][v]['function'] == 2],
                arrowstyle='-[',
                width = scaled_weights[1],
                edge_color=colors[1],
                arrowsize=10,
                alpha=0.8
            )

            # Draw labels
            nx.draw_networkx_labels(network, pos, font_size=12)

            plt.suptitle(name)
            plt.savefig(name + '.png')
            plt.close()

    def getSizes(self, G):

        max_width = 30  # Maximum width in inches
        max_height = 18  # Maximum height in inches

        # Calculate the figure size based on the number of nodes
        num_nodes = G.number_of_nodes()
        fig_width = min(num_nodes * 1.5, max_width)  # Limit width
        fig_height = min(num_nodes * 1, max_height)
        fig_size = (fig_width, fig_height)

        # Calculate node size based on the number of nodes
        node_size = min(2500, 2000 + (num_nodes * 20))  # Limit node size
        return node_size, fig_size


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class MotifTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, motif):
        """
        Inserts a motif into the trie.
        Motif should be sorted to maintain consistency.
        """
        current = self.root
        for edge in sorted(motif):
            if edge not in current.children:
                current.children[edge] = TrieNode()
            current = current.children[edge]
        current.is_end = True

    def is_submotif(self, motif):
        """
        Checks if the given motif is a sub-motif of an existing motif in the trie.
        """
        current = self.root
        for edge in sorted(motif):
            if edge not in current.children:
                return False
            current = current.children[edge]
        return current.is_end























