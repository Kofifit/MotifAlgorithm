import argparse
from time import time
from util_functions import UtilFunctions
from Classes import NetworkDisessembler, NetworkDeltaExtractor, DeltaNetworkMotifAnalyzer, GraphVisualization, MotifSearcher


def run(solutions_file, algorithm_type, n, motifs_file, output_file, s_t):

    print('Started analysis...')
    motifSearcher = False
    if motifs_file != '':
        motifSearcher = MotifSearcher(motifs_file, n)

    if s_t == 'FULL':
        solutions = UtilFunctions.excel2solutionSetList(solutions_file)
        analyses_full = []
        time_full = []
        for i, sol in enumerate(solutions):
            start_time = time()
            # Perform motif analysis the original solution
            analyzer = DeltaNetworkMotifAnalyzer(sol, n, algorithm_type)
            analysis = analyzer.originAnalysis
            end_time = time()
            elapsed_time = end_time - start_time
            time_full.append(elapsed_time)
            # Save analysis results to CSV files
            filename = output_file + f'{i}.csv'
            analyses_full.append(analysis)
            analyzer.saveAnalysis(analysis, filename)

        print('Ended analysis')
        ave_time = sum(time_full) / len(time_full)
        print(f"\n##ALGORITHM USED {algorithm_type}##")
        print(f'Average time full analysis took was {ave_time} seconds')
        print('Time for each full analysis below:')
        print(time_full)

    else:
        solutions = UtilFunctions.excel2solutionSetList(solutions_file)
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

        # print('Creating visualizations...')
        # getGraphs(solutions, analyses_modified)
        # print('Finished creating visualizations')


def  getGraphs(solutions, analyses_lst):

    folder = 'Figures/'

    ## Draw combination graph for all solutions
    GraphDrawer = GraphVisualization(solutions)
    GraphDrawer.createCombinationGraph(solutions, folder + f"Graph-Combined_solutions")
    for i, s in enumerate(solutions):
        GraphDrawer.createDeltaNetworkGraph(s, folder + f"solution{i}_delta_graph")
        GraphDrawer.createRegularGraph(s, folder + f"solution{i}_regular_graph")
        motifs_df = analyses_lst[i]
        new_network = UtilFunctions.addMotifs2Network(s, motifs_df)
        for motif_index, m in motifs_df.iterrows():
            GraphDrawer.createMotifDeltaNetworkGraph(new_network, folder + f"solution{i}_motif{motif_index}_delta_motif_graph", motif_index)
            GraphDrawer.createMotifNetworkGraph(new_network, folder + f"solution{i}_motif{motif_index}_regular_motif_graph", motif_index)


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

    # solutions_file = 'Solutions/network1_modified.xlsx'
    # algorithm_type = 'Nauty'
    # n = 3
    # motifs_file = ''
    # output_file = 'analysis'
    run(solutions_file, algorithm_type, n, motifs_file, output_file,'FULL')
