import argparse
from time import time
import pandas as pd
from util_functions import UtilFunctions
from Classes import NetworkDisessembler, NetworkDeltaExtractor, DeltaNetworkMotifAnalyzer, GraphVisualization, MotifSearcher
from DataTransformer import DataTransformer

def run(solutions_file, algorithm_type, n, motifs_file, output_file, X):

    print('Started analysis...')
    motifSearcher = False
    if motifs_file != '':
        motifSearcher = MotifSearcher(motifs_file, n)

    dataTransformer = DataTransformer(solutions_file)
    solutions, gene_map = dataTransformer.Transform()
    # if s_t == 'FULL':
    #     analyses_full = []
    #     time_full = []
    #     for i, sol in enumerate(solutions):
    #         start_time = time()
    #         # Perform motif analysis the original solution
    #         analyzer = DeltaNetworkMotifAnalyzer(sol, n, algorithm_type)
    #         analysis = analyzer.originAnalysis
    #         end_time = time()
    #         elapsed_time = end_time - start_time
    #         time_full.append(elapsed_time)
    #         # Save analysis results to CSV files
    #         filename = output_file + f'{i}.csv'
    #         analyses_full.append(analysis)
    #         analyzer.saveAnalysis(analysis, filename)
    #
    #     print('Ended analysis')
    #     ave_time = sum(time_full) / len(time_full)
    #     print(f"\n##ALGORITHM USED {algorithm_type}##")
    #     print(f'Average time full analysis took was {ave_time} seconds')
    #     print('Time for each full analysis below:')
    #     print(time_full)

    analyses_modified = []
    time_modified = []
    for i, sol in enumerate(solutions):
        if i == 0:
            # Perform motif analysis the original solution
            analyzer = DeltaNetworkMotifAnalyzer(sol, n, algorithm_type)
            analysis = analyzer.originAnalysis
        else:
            # Extract delta network from modified solutions
            extractor = NetworkDeltaExtractor(n, sol)
            extractor.extractDeltaNetwork()
            delta = extractor.getDeltaNetwork()
            deltaNetwork = NetworkDisessembler(delta).getNetwork()
            # Perform motif analysis on the delta network
            start_time = time()
            analysis = analyzer.analyze(deltaNetwork)
            analysis = analyzer.compare(sol, deltaNetwork, analysis)
            end_time = time()
            elapsed_time = end_time - start_time
            time_modified.append(elapsed_time)

        # Search for motifs
        if motifSearcher:
            analysis = motifSearcher.findMotifs(analysis)

        # Save analysis results to CSV files
        filename = output_file + f'{i}.csv'
        analyses_modified.append(analysis)
        analyzer.saveAnalysis(analysis, filename)

    print('Ended analysis')
    ave_time = sum(time_modified) / len(time_modified)
    print(f"\n##ALGORITHM USED {algorithm_type}##")
    print(f'Average time modified analysis took was {ave_time} seconds')
    print('Time for each modified analysis below:')
    print(time_modified)

    gene_map_swapped = {value: key for key, value in gene_map.items()}
    for s in solutions:
        for t, edge in s.items():
            edge[0] = [gene_map_swapped.get(value, value) for value in edge[0]]

    # print('Creating visualizations...')
    # visualizer = GraphVisualization(solutions, analyses_modified)
    # visualizer.getGraphs(X)
    # print('Finished creating visualizations')

    gene_map_df = pd.DataFrame(list(gene_map_swapped.items()), columns=['Gene Number', 'Gene Name'])
    gene_map_df.to_csv(f'Analyses/gene_dictionary{X}.csv', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that gets s (solutions file), a (algorithm type), n (motif size), "
                    "m (motifs file), o (output file)"
    )
    parser.add_argument("--s", required=True, type=str)
    parser.add_argument("--a", required=True, type=str)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--m", required=False, type=str)
    parser.add_argument("--o", required=False, type=str)
    args = parser.parse_args()

    solutions_file = args.s
    algorithm_type = args.a
    n = args.n
    motifs_file = ''
    if args.m:
        motifs_file = args.m
    output_file = 'Analyses/analysis'
    if args.o:
        output_file = args.o

# if __name__ == "__main__":
#     algorithm_type = 'Nauty'
#     n = 3
#     motifs_file = ''
#     for i in range(1, 22):
#         print(f'### ANALYSIS {i} ###')
#         solutions_file = f'Solutions/network{i}.xlsx'
#         output_file = f'Analyses/analysis{i}'
#         run(solutions_file, algorithm_type, n, motifs_file, output_file, i)
