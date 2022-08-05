I thought I'd start off with some brief notes on ```run_alternative_data.plot_alt_data.py```, since that's most important. Below the notes is more information on the directory.
# Notes on ```plot_alt_data.py```
> The methods FROM assign_band_curves(interval) TO calc_sys_error_original(...), inclusive, are all for calculating error bands.

> plot_data_set(...) plots the asymmetry data and Alarcon and Weiss along with a given data set's Gramolin model onto a set of axes (representing a panel)

> plot_subplot(...) is still for a specific axes, it calls plot_data_set on that axes and adds essentially all the settings for each panel. There's a bunch of pre-defined parameters to adjust these settings. main() calls this method for each data set and puts in the settings. main() also contains whole-figure settings. See comments. *(draw_model_handle(ax, pos, size) is a helper for plot_subplot(...))
# More information
You should be able to ignore the following folders:
> ```main_working_folder```, contains my draft work 
> 
> ```fun_homework```, contains the first homework code

And you won't need these folders, although they do have some useful reference material:
> ```misc_ref```, contains some miscellaneous references including Alarcon code, articles, data files 
> 
> ```original_gramolin_code```, contains the original code from Gramolin

The ```run_alternative_data``` folder contains the important stuff.

For the files directly in the directory root:

First, ```alt_data_methods.py``` and ```run_alt_data.py``` are essentially just the supplementary material code, so you shouldn't need to do anything with these, I would just go to the supplementary material if you want to look at these (I made a few polishing updates like comments and combined the two into one file).

The rest of the files are the meat:
> ```plot_alt_data.py``` is the most important file, it makes the plot (mostly my own code so not as clean). 
> 
> ```modified_fit.py``` and ```modified_plot.py``` are modified versions of Gramolin's code which are used in  the other scripts. ```models.py``` is Gramolin's code which is also used in other scripts.
> 
> ```asym_chi2.py``` runs the chi^2 tests for the models against the asymmetry data.

Important note: the only methods from ```modified_plot.py``` that ```plot_alt_data.py``` uses are calc_params(...), which is purely convenient and the same as the original ```plot.py```, and plot_ge_gm(...), which is heavily modified. Without the ratio distribution (which is not needed in the current implementation except for checking), it is mostly just commenting stuff out of the original method. This could probably be refactored as a method for ```plot_alt_data.py``` itself easily, but I just have it implemented like this for now. ```modified_fit.py``` is just used by everyone for making sure the other data sets run and other like insubstantial convenience changes, I think. 

For the folders in the directory, you can ignore most of them, here are what they contain:
> ```assorted_testing``` contains random side things like running just PRad data or picking the model by the asym chi^2
> 
> ```asymdata``` and ```data``` have the data files
> 
> ```figures``` and ```output_texts``` contain those things
> 
> ```misc``` is just old or random things
