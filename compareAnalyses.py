import pandas as pd

if __name__ == "__main__":

    file_type = ['', '_modified']
    for i in range(1, 11):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$4')
        print(f'################################# NETWORK {i} #################################')
        for s in range(0, 7):
            print(f'################################# SOLUTION {s} #################################')
            full_fn = f'Analyses/CHANGEDanalysis{i}_{file_type[0]}{s}.csv'
            mod_fn = f'Analyses/CHANGEDanalysis{i}_{file_type[1]}{s}.csv'
            full_df = pd.read_csv(full_fn)
            mod_df = pd.read_csv(mod_fn)
            for row_num, row_full in full_df.iterrows():
                # Find row of matching motif in modified analysis
                row_num_mod = mod_df['Motif'][mod_df['Motif'] == row_full['Motif']].index
                if row_num_mod.empty:
                    print('#################### MISSING MOTIF ####################')
                    print(full_fn)
                    print(row_num)
                else:
                    row_mod = mod_df.iloc[row_num_mod[0]]
                    if row_mod['Number of appearances in network'] != row_full['Number of appearances in network']:
                        print('#################### ISSUE with appearances number ####################')
                        print(row_mod['Number of appearances in network'])
                        print(row_full['Number of appearances in network'])
                    for edge in row_mod['Edges indices']:
                        if edge not in row_full['Edges indices']:
                            print('#################### ISSUE edge does not exist in full analysis ####################')
                            print(edge)
                    for edge in row_full['Edges indices']:
                        if edge not in row_mod['Edges indices']:
                            print('#################### ISSUE edge does not exist in modified analysis ####################')
                            print(edge)
                    for location in row_mod['Location of appearances in network']:
                        if location not in row_full['Location of appearances in network']:
                            print('#################### ISSUE location does not exist in full analysis ####################')
                            print(location)
                    for location in row_full['Location of appearances in network']:
                        if location not in row_mod['Location of appearances in network']:
                            print('#################### ISSUE location does not exist in modified analysis ####################')
                            print(location)

                # Select same columns from CSV2
                # # Check if both DataFrames are equal
                # for u, val in enumerate(csv1_columns):
                #     if len(val) != csv1['Number of appearances in network'][u]:
                #         print(len(val))
                #         print(csv1['Number of appearances in network'][u])
                #         print('XXXXXXXXXXXXX')
                #     if len(csv2_columns[u]) != csv2['Number of appearances in network'][u]:
                #         print(len(csv2_columns[u]))
                #         print(csv2['Number of appearances in network'][u])
                #         print('XXXXXXXXXXXXX')
                #     if len(val) != len(csv2_columns[u]):
                #         print('################ ERROR ################')
                #         print(len(val))
                #         print(len(csv2_columns[u]))


