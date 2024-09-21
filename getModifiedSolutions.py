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

    solutions_file = 'Solutions/network1.xlsx'
    output_file = 'Solutions/network1_modified.xlsx'
    solutions = UtilFunctions.excel2solutionSetList(solutions_file)

    col_names = ['edge', 'Activation/Repression', 'delta']

    UtilFunctions.solutionSetModified2excel(solutions, col_names, output_file)

    print('Solutions saved successfully!')
