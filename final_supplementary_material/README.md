# Comment on "Transverse charge density and the radius of the proton"

This repository contains supplementary material to 'Comment on transverse charge density and the radius of the proton', a comment on the article
> A. V. Gramolin and R. L. Russell, "Transverse charge density and the radius of the proton", [Phys. Rev. D **105**, 054004 (2022)](https://doi.org/10.1103/PhysRevD.105.054004).

Extractions of the charge radius for all data sets can be completed using the ```different_input_sets.py``` script. The ```different_input_sets.py``` script runs the extractions by referencing the ```modified_fit.py``` script, which is a modified version of the ```fit.py``` script from Gramolin and Russell's PYTHON code. All modifications are recorded in the text file ```edits.txt```.

Instructions for running ```different_input_sets.py```:
> In ```main()```, each data set is run by calling ```print_table_replications(max_order, lambdas, data_filepath, data_file_name)```.

> ```lambdas``` should be a 1-D iterable (e.g. list, numpy array) containing the values to be scanned when searching for the optimal regularization parameter. 

> ```max_order``` is the maximum order model to be fit, ```data_file_path``` and ```data_file_name``` are the path and name of the data file. Note that the max_order should not exceed 8, as the original PYTHON code does not support this (and we have not changed it).

> Note on run time: the run time of the script will increase with the size of ```lambdas``` and the value of ```max_order```. The implementation in this repository has a ```max_order``` of 8 and 151 elements in ```lambdas```, and will take hours on most machines.

> Output will be printed into a .txt file called ```output.txt```

The folder called 'gramolin_repo' is an exact copy of the repository containing Gramolin and Russell's original PYTHON code, which can be found at: https://github.com/gramolin/radius. The folder is included for reference and for one file/module, plot.py, which is imported and utilized by the modified_fit.py script. 
