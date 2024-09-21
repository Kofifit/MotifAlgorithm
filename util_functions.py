import math
import pandas as pd
import numpy as np
import random
import copy
import networkx as nx
import re

class UtilFunctions:

    @staticmethod
    def csv2network(filename):
        '''
        Converts a CSV file containing network data into a dictionary representing the network.

        Parameters:
            filename (str): The name of the CSV file containing the network data.

        Returns:
            dict: A dictionary representing the network.
        '''
        try:
            df = pd.read_csv(filename)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            return UtilFunctions.df2network(df)
        except Exception as error:
            return error


    @staticmethod
    def network2csv(filename, network, col_names):
        '''
        Converts a dictionary representing a network into a CSV file.

        Parameters:
            filename (str): The name of the CSV file to save the network data.
            network (dict): A dictionary representing the network.
            col_names (list): A list of column names for the CSV file.
        '''
        try:
            df = UtilFunctions.network2df(network, col_names)
            df.to_csv(filename, sep=',')
        except Exception as error:
            return error

    @staticmethod
    def network2txt(filename, network, col_names):
        '''
        Converts a dictionary representing a network into a CSV file.

        Parameters:
            filename (str): The name of the CSV file to save the network data.
            network (dict): A dictionary representing the network.
            col_names (list): A list of column names for the CSV file.
        '''
        try:
            df = UtilFunctions.network2df(network, col_names)
            df.drop(columns=['delta'], inplace=True)
            df.to_csv(filename, sep=' ', header=False, index=False)
        except Exception as error:
            return error


    @staticmethod
    def df2network(df):
        '''
        Converts a DataFrame into a dictionary representing the network.

        Parameters:
            df (DataFrame): The DataFrame containing network data.

        Returns:
            dict: A dictionary representing the network.
        '''
        df = df.astype(int)
        network = dict()
        for index, row in df.iterrows():
            try:
                network[int(row['Index'])] = [[int(row['Gene A']), int(row['Gene B'])], int(row['Activation/Repression']), int(row['delta'])]
            except:
                network[int(row['Index'])] = [[int(row['Gene A']), int(row['Gene B'])], int(row['Activation/Repression'])]


        return network


    @staticmethod
    def network2df(network, col_names):
        '''
        Converts a dictionary representing a network into a DataFrame.

        Parameters:
            network (dict): A dictionary representing the network.
            col_names (list): A list of column names for the DataFrame.

        Returns:
            DataFrame: The DataFrame containing network data.
        '''

        df_origin = pd.DataFrame.from_dict(network, orient='index', columns=col_names)
        df_genes = pd.DataFrame(df_origin.edge.tolist(), index=df_origin.index, columns=['Gene A', 'Gene B'])
        df_origin.drop(columns=['edge'], inplace=True)
        df = pd.concat([df_genes, df_origin], axis=1, join="inner")
        df.sort_index()
        return df


    @staticmethod
    def addMotifs2Network(network, motifs_df):
        '''
        Adds motifs data to a network dictionary.

        Parameters:
            network (dict): The network dictionary to which motifs will be added.
            motifs_df (DataFrame): DataFrame containing motifs data.

        Returns:
            dict: The updated network dictionary with motifs added.
        '''
        for i, motif in motifs_df.iterrows():
            indices = set(sum(motif['Edges indices'], ()))
            for key in indices:
                network[key].append(i)
        return network


    @staticmethod
    def csv2analysis(filename):
        '''
        Converts a CSV file containing motif analysis data into a DataFrame.

        Parameters:
            filename (str): The name of the CSV file containing the analysis data.

        Returns:
            DataFrame: The DataFrame containing motif analysis data.
        '''
        try:
            df = pd.read_csv(filename)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            return UtilFunctions.analysis2df(df)
        except Exception as error:
            return error

    @staticmethod
    def analysis2df(analysis):
        '''
        Converts a DataFrame containing motif analysis data into a specific format.

        Parameters:
            analysis (DataFrame): The DataFrame containing analysis data.

        Returns:
            DataFrame: The modified DataFrame with analysis data.
        '''
        new_indices = []
        indices = analysis['Edges indices']
        for i in indices:
            numbers = re.findall(r'\d+', i)
            new_indices.append([int(x) for x in numbers])
        analysis['Edges indices'] = new_indices
        return analysis


    @staticmethod
    def solutionSetFull2excel(solution_set, col_names, filename):
        '''
        Writes a set of solutions to an Excel file.

        Parameters:
            solution_set (list): A list of DataFrames, each containing a solution.
            col_names (list): A list of column names for the Excel file.
            filename (str): The name of the Excel file to save the solutions.
        '''
        with pd.ExcelWriter(filename) as writer:
            for index, solution in enumerate(solution_set):
                df = UtilFunctions.network2df(solution, col_names=col_names)
                sheet_name = '#' + str(index)
                df.to_excel(writer, sheet_name=sheet_name)


    @staticmethod
    def solutionSetModified2excel(solution_set, col_names, filename):
        '''
        Writes a set of modified solutions to an Excel file.

        Parameters:
            solution_set (list): A list of DataFrames, each containing a modified solution.
            col_names (list): A list of column names for the Excel file.
            filename (str): The name of the Excel file to save the modified solutions.
        '''
        with pd.ExcelWriter(filename) as writer:
            origin = solution_set[0]
            for index, solution in enumerate(solution_set):
                UtilFunctions.find_delta(origin, solution)
                df = UtilFunctions.network2df(solution, col_names)
                sheet_name = '#' + str(index)
                df.to_excel(writer, sheet_name=sheet_name)


    @staticmethod
    def excel2solutionSetList(filename):
        '''
        Reads solutions from an Excel file and returns them as a list of DataFrames.

        Parameters:
            filename (str): The name of the Excel file containing the solutions.

        Returns:
            list: A list of dictionaries representing the solutions (each solution is a network).
        '''
        solutions = []
        df_dict = pd.read_excel(filename, sheet_name=None)
        for key in df_dict.keys():
            df = df_dict[key]
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            solution = UtilFunctions.df2network(df)
            solutions.append(solution)
        return solutions

    @staticmethod
    def edge2df(edge, df):
        '''
        Adds an edge to a DataFrame.

        Parameters:
            edge (tuple): The edge to be added to the DataFrame.
            df (DataFrame): The DataFrame to which the edge will be added.

        Returns:
            DataFrame: The updated DataFrame with the added edge.
        '''
        index = edge[0]
        nodes = edge[1][0]
        func = edge[1][1]
        df.loc[len(df.index)] = [index, nodes[0], nodes[1], func]
        return df

    @staticmethod
    def  generateRandSolutionSet(network, solution_number, network_size, delta_size):
        '''
        Generates a set of random solutions for a given network.

        Parameters:
            network (dict): A dictionary representing the network.
            solution_number (int): The number of solutions to generate.
            network_size (int): The size of the network.
            delta_size (float): The delta size for generating solutions.

        Returns:
            list: A list of dictionaries, each representing a random solution.
        '''
        solutions_set = []
        origin_keys = []
        rest_keys = []
        counter = 0
        while counter < solution_number:
            counter += 1
            if counter == 1:
                keys = random.sample(list(network.keys()), network_size)
                origin_keys = keys
                for k in network.keys():
                    if k not in origin_keys:
                        rest_keys.append(k)
            else:
                keys_from_origin = random.sample(origin_keys, math.ceil(network_size*(1-delta_size)))
                keys_from_rest = random.sample(rest_keys, math.floor(network_size*delta_size))
                keys = keys_from_origin + keys_from_rest

            solution_temp = {k: network[k] for k in keys if k in network}
            solutions_set.append(solution_temp)
        return solutions_set


    @staticmethod
    def find_delta(originNetwork, otherNetwork):
        '''
        Finds the differences between two examples-input and updates the second network accordingly.

        Parameters:
            originNetwork (dict): The original network.
            otherNetwork (dict): The network to be updated with differences.
        '''

        originNetworkCopy = copy.deepcopy(originNetwork)

        if len(otherNetwork[list(otherNetwork.keys())[0]]) == 2:
            UtilFunctions.add_delta(otherNetwork)

        for key in otherNetwork.keys():
            if key in originNetworkCopy:
                del originNetworkCopy[key]
                d = 0
            else:
                d = 1
            otherNetwork[key][-1] = d

        for key in originNetworkCopy.keys():
            value = originNetworkCopy[key]
            d = -1
            value[-1] = d
            otherNetwork[key] = value


    @staticmethod
    def add_delta(network):
        for key in network.keys():
            network[key].append(0)

    @staticmethod
    def Network2NetworkX(network):
        graph = nx.DiGraph()
        for edge in network.values():
            motifs = []
            if len(edge) > 3:
                motifs = [m for m in edge[3:]]
            graph.add_edge(edge[0][0], edge[0][1], weight=edge[-1], function=edge[1], delta=edge[2], motifs=motifs)
        return graph

    @staticmethod
    def CombineSolutions(solutions):
        merged_solutions = {}

        # Determine the frequency of occurrence for each interaction across all solutions
        for s in solutions:
            for key, edge in s.items():
                if key in merged_solutions.keys():
                    new_val = merged_solutions.get(key)
                    new_val[-1] += 1
                else:
                    new_val = copy.deepcopy(edge)
                    new_val.append(1)
                merged_solutions[key] = new_val

        # Normalize frequency value to 0-1 scale
        for key, val in merged_solutions.items():
            val[-1] /= len(solutions)
            merged_solutions[key] = val

        return merged_solutions

    @staticmethod
    def get_unique_nodes(vector):
        # This function get a vector of integers
        # It returns the unique values in the vector, in the order they appeared
        # For example: vector = [1,0,0,2] --> the function returns [1,0,2]

        output = []
        for val in vector:
            if val not in output or not output:
                output.append(val)
        return output

    @staticmethod
    def get_label_dict(old_labels, new_labels):
        # This function get two lists - old_labels of nodes and new_labels of nodes
        # It returns a dictionary that matches the old labels to the new labels

        label_dict = dict()
        for i, old in enumerate(old_labels):
            label_dict[old] = new_labels[i]
        return label_dict

    @staticmethod
    def get_all_possible_edges(n):
        # This function get an integer n as the number of nodes
        # It returns a dictionary with all possible edges with n nodes
        # ** NOTE all returned edges are positive ([[node, node], positive(1)/negative(2) ]

        # Define dict to store all possible edges
        all_edges = dict()
        # Loop to find all possible edges
        index = 0
        for n1 in range(0, n):
            for n2 in range(0, n):
                if n1 != n2:
                    all_edges[index] = [[n1, n2], 1]
                    index += 1
        return all_edges

    @staticmethod
    def get_graph(combo, network):
        # This function get a list of edges
        # It returns a subgraph as a dict and as a Graph as well as a list of all nodes in the subgraph

        subgraph = dict()
        g = Graph()
        nodes_subgraph = []
        for edge in combo:
            edge = network[edge][0]
            if edge[0] not in subgraph:
                subgraph[edge[0]] = []
            subgraph[edge[0]].append(edge[1])  # Save subgraph
            g.addEdge(edge[0], edge[1])  # Save subgraph in Graph class for DFS algorithm
            nodes_subgraph.extend([edge[0], edge[1]])
        return subgraph, g, np.array(nodes_subgraph)

    @staticmethod
    def get_buffer_graph(combo, subgraph):
        # This function get a list of new labels for subgraph's nodes and a subgraph
        # It returns a buffer_subgraph with new labeled nodes

        buffer_subgraph = dict()
        buffer_subgraph['Matched'] = True
        for key, val in subgraph.items():
            if key != 'Matched':
                buffer_subgraph[combo[key]] = sorted([combo[v] for v in val])
        return buffer_subgraph

    @staticmethod
    def has_n_nodes(n, nodes_subgraph):
        # This function get a list of nodes and a list of the nodes in a subgraph
        # It returns True if subgraph has all nodes and False otherwise

        nodes_subgraph = np.unique(nodes_subgraph)
        nodes_num = len(nodes_subgraph)
        if np.array_equal(n, nodes_num):
            return True
        else:
            return False

    @staticmethod
    def is_fully_connected(graph, start_node=0):
        # This function get a Graph and checks if it is fully connected with DFS algorithm
        # It returns True if the graph is fully connected and False otherwise

        return graph.DFS(start_node)

    @staticmethod
    def is_subgraph_found(subgraph_indices, all_indices):

        for graph in all_indices:
            if len(graph) > len(subgraph_indices):
                if set(subgraph_indices).issubset(set(graph)):
                    return True
        return False











