sudo apt update
sudo apt install python3
sudo apt install python3-pip
sudo apt install python3-virtualenv
virtualenv rein_motif_algorithm_env
source rein_motif_algorithm_env/bin/activate
pip3 install argparse numpy pandas networkx matplotlib openpyxl
sudo apt install graphviz graphviz-dev -y
pip install pygraphviz
cd gtrieScannerFolder
make
cd ..
source rein_motif_algorithm_env/bin/activate
