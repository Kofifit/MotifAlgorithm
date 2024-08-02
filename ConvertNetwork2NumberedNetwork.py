import pandas as pd

class Converter:

    def __init__(self, filename, new_filename, type):
        self.filename = filename
        self.new_filename = new_filename
        self.type = type

    def convert(self):

        if self.type =='numberFunction':
            self.numberedConverter()
        elif self.type == 'symbolFunction':
            self.symboledConverter()

    def numberedConverter(self):
        # Get network from csv file in dataframe
        df = pd.read_csv(self.filename)

        # Get names of all genes and number them in a dictionary
        gene_dict = {}
        function_dict = {}
        counter_g = 1
        counter_f = 1
        for index, row in df.iterrows():
            for n in range(0, 2):
                name = row.iloc[n]
                if name not in gene_dict.keys():
                    gene_dict[name] = counter_g
                    counter_g += 1
            function = row.iloc[2]
            if function not in function_dict.keys():
                function_dict[function] = counter_f
                counter_f += 1

        # Replace gene names to numbers in df
        for index, row in df.iterrows():
            for n in range(0, 2):
                df.iat[index, n] = gene_dict[df.iat[index, n]]
            df.iat[index, 2] = function_dict[df.iat[index, 2]]

        # Save changes into new csv file
        df.to_csv(self.new_filename, index=False)

    def symboledConverter(self):

        gene_dict = {}
        function_dict = {}
        counter_g = 1
        counter_f = 1
        source_lst = []
        target_lst = []
        function_lst = []
        delta_lst = []
        # Get network from csv file in dataframe
        with open(self.filename, 'r') as f:
            for line in f:
                if line.startswith('$'):
                    break
                line = line.strip().split()

                function = line[1]
                num = False
                if function == '->':
                    num = 1
                elif function == '-|':
                    num = 2


                if num:
                    function_lst.append(num)
                    delta_lst.append(0)
                    source = line[0]
                    if source not in gene_dict.keys():
                        gene_dict[source] = counter_g
                        counter_g += 1
                    source_lst.append(gene_dict[source])

                    target = line[2]
                    if target not in gene_dict.keys():
                        gene_dict[target] = counter_g
                        counter_g += 1
                    target_lst.append(gene_dict[target])

        # Save changes into new csv file
        df = pd.DataFrame()
        df['Gene A'] = source_lst
        df['Gene B'] = target_lst
        df['Activation/Repression'] = function_lst
        df['delta'] = delta_lst
        df.to_csv(self.new_filename)

def main():
    for i in range(1, 11):
        filename = f'networks/model{i}.txt'
        new_filename = f'networks/network{i}.txt'
        converter = Converter(filename.strip(), new_filename.strip(), 'symbolFunction')
        converter.convert()

if __name__ == '__main__':
    main()


