There are two main directories: ```final_supplementary_material``` and ```working_folder```.

# ```final_supplementary_material```
This contains the code and files that are intended for the supplementary material (which we discussed) -- so just the things needed for running the different input data sets. This directory has it's own README, which you should refer to for the contents of that directory.

# ```working_folder```
This directory contains most of the work I did. 

You should be able to ignore the following folders:
> ```main_working_folder```, contains my draft work
> ```fun_homework```, contains the first homework code

And you won't need these folders, although they do have some useful reference material:
> ```misc_ref```, contains some miscellaneous references including Alarcon code, articles, data files 
> ```original_gramolin_code```, contains the original code from Gramolin

The ```run_alternative_data``` folder contains the important stuff.

For the files directly in the directory:

First, ```alt_data_methods.py``` and ```run_alt_data.py``` are essentially just the supplementary material code, so you shouldn't need to do anything with these, I would just go to the supplementary material if you want to look at these (I made a few polishing updates like comments and combined the two into one file).

The rest of the files are the meat:
> ```plot_alt_data.py``` is the most important file, it makes the plot (mostly my own code so might not be as clean)
> ```modified_fit.py``` and ```modified_plot.py``` are modified versions of Gramolin's code which are used in  the other scripts. ```models.py``` is Gramolin's code which is also used in other scripts.
> ```asym_chi2.py``` runs the chi^2 tests for the models against the asymmetry data.



