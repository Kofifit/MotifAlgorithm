This is a project of a motif lookup algorithm.
Our intention and goal is to provide further analysis to the analysis provided by the REIN algorithm.
The purpose for these analyses is to help conclude which solution is more likely in the real world.

We suggest a delta representation of the solutions. One solution would e defined as the origin and the rest would be defined as its derivatives. Each interaction in the derivatives solutions would be marked with a delta (0 if the interaction also exists in both the origin solution and the current solution, -1 if the exists only in the origin solution and not in the current solution and 1 if the interaction exists in the current solution but not in the origin solution). Due to the fact that the differences between the solutions are not too great, we can investigate only these differences. Instead of analyzing each solution fully and completing redundant computations that were done in other solutions as well. This helps reduce running time significantly. Only the origin solution would be analyzed fully, while in the rest of the solutions only the differences would be analyzed. 

The algorithm provides the following analyses for the solutions sets obtained from the REIN algorithm - 
1. Visual analysis that includes graphs that help interpret the data (A graph of the solutions combined, A graph of each motif in the solution, A graph of the delta network, A graph of the delta network with highlight on the motif). 
2. A motif analysis for each solution separately. This analysis contains the motifs that appeared in the network, their locations and how many times they appeared.
3. A tool for motif lookup in the solutions. You can use this to upload a file that contains the motif you are interested in. The analysis would only provide the motifs you searched for. An example for the file is in 'motifSearcherInputExample.txt' in the 'examples-input' folder.

The motif analysis is done using Brute-Force algorithm or Nauty algorithm (obtained from gTries scanner ** ADD CITATION **).
The tool was only tested on Ubuntu.

To install the tool and all necessary packages run the following script through the terminal - 'install.sh'

After installing in the same terminal run the following lines to load the environment :
```source rein_motif_algorithm_env/bin/activate```

To run the algorithm you can run the following line in your terminal:
```python3 main.py --s /location/of/your/solutions/file.xlsx --a Nauty (algorithm name, another option is BruteForce) --n 3 (size of motifs up to 8)```

The 'main.py' file receives the following parameters - s (solutions file), a (algorithm type), n (motif size), *m (motifs file), *o (output file)
* These are optional parameters



