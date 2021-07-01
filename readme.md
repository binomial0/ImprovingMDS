
# Improving MDS

This is the accompanying code to my bachelor's thesis
"Improving the Computational Efficiency of Metric Multidimensional Scaling".

The file `main.py` contains the MDS solvers that were compared in the thesis.
The file `figures.py` contains the code used to generate the figures, which is
less mature. It imports `mod_algorithms.py`, which duplicates two optimisation
algorithms from scipy and scikit-learn, and modifies them to output some
data that is needed for the plots.