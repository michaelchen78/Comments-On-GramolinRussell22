[![arXiv](https://img.shields.io/badge/arXiv-2102.13022-b31b1b.svg)](https://arxiv.org/abs/2102.13022)
[![DOI:10.1103/PhysRevD.105.054004](https://img.shields.io/badge/DOI-10.1103/PhysRevD.105.054004-0000ff.svg)](https://doi.org/10.1103/PhysRevD.105.054004)

# Proton radius extraction from transverse charge density

This repository contains all of the code and data necessary to generate the full analysis and plots of our paper:
> A. V. Gramolin and R. L. Russell, "Transverse charge density and the radius of the proton", [Phys. Rev. D **105**, 054004 (2022)](https://doi.org/10.1103/PhysRevD.105.054004)

All of the data analyzed come from the [A1 Collaboration at Mainz](https://wwwa1.kph.uni-mainz.de), described in:
> J. C. Bernauer *et al.* (A1 Collaboration), [Phys. Rev. Lett. **105**, 242001 (2010)](https://doi.org/10.1103/PhysRevLett.105.242001) <br>
> J. C. Bernauer *et al.* (A1 Collaboration), [Phys. Rev. C **90**, 015206 (2014)](https://doi.org/10.1103/PhysRevC.90.015206)

### Requirements

Our analysis scripts require Python 3.6 or above. The Python package dependencies can be installed via:
```
> pip3 install -r requirements.txt
```
### Running the code

The `fit.py` script runs the full analysis described in the paper including cross-validation. You can change the model order and regularization parameter from the default "best-fit" ones by command line arguments. Use the `--help` flag to see how to pass the arguments:
```
> python3 fit.py --help
usage: fit.py [-h] [--order ORDER] [--reg_param REG_PARAM]

Fit and validate models to cross section data.

optional arguments:
  -h, --help              show this help message and exit
  --order ORDER           order of form factor expansion (default: N=5)
  --reg_param REG_PARAM   regularization parameter (default: lambda=0.02)
```
You can regenerate all of the plots used in the paper and supplemental material by running the `plot.py` script.
