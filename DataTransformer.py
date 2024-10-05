import pandas as pd
from util_functions import UtilFunctions
import argparse

class DataTransformer:

    def __init__(self, filename):
        self.filename = filename

    def Transform(self):
        solutions = []
        gene_map = {}
        counter = 1
        df_dict = pd.read_excel(self.filename, sheet_name=None)
        for key in df_dict.keys():
            df = df_dict[key]
            genes = pd.concat([df['Gene A'], df['Gene B']]).unique()
            for g in genes:
                if g not in gene_map:
                    gene_map[g] = counter
                    counter += 1
            df['Gene A'] = df['Gene A'].map(gene_map)
            df['Gene B'] = df['Gene B'].map(gene_map)
            df['delta'] = [0] * len(df)
            df.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)
            solutions.append(df)

        for i, s in enumerate(solutions):
            temp_s = UtilFunctions.df2network(s)
            print(len(temp_s))
            if i == 0:
                origin = temp_s
            UtilFunctions.find_delta(origin, temp_s)
            solutions[i] = temp_s

        return solutions, gene_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that gets f (solutions file) o (new filename for output)"
    )
    parser.add_argument("--f", required=True, type=str)
    args = parser.parse_args()

    solutions_file = args.f
    dataTransformer = DataTransformer(solutions_file.strip())
    solutions, gene_map = dataTransformer.Transform()
    for s in solutions:
        print(s)

