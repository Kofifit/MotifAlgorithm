import argparse
from util_functions import UtilFunctions

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Script that gets f (solutions file) o (new filename for output)"
    # )
    # parser.add_argument("--f", required=True, type=str)
    # parser.add_argument("--o", required=True, type=str)
    # args = parser.parse_args()
    #
    # solutions_file = args.f
    # output_file = args.o
    for i in range(1, 11):
        solutions_file = f'Solutions/network{i}.xlsx'
        output_file = f'Solutions/network{i}_modified.xlsx'
        solutions = UtilFunctions.excel2solutionSetList(solutions_file)

        col_names = ['edge', 'Activation/Repression', 'delta']

        UtilFunctions.solutionSetModified2excel(solutions, col_names, output_file)

        print('Solutions saved successfully!')
