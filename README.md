# PerturbNet
This software implments the statistical methods for learning gene networks underlying clinical phenotypes using SNP perturbations, described in the following paper:

McCarter, C., Howrylak, J., & Kim, S. (2020). Learning gene networks underlying clinical phenotypes using SNP perturbation. PLoS computational biology, 16(10), e1007940.

The statistical model and optimization method were previously described in the following machine learning conference papers:

McCarter, C., & Kim, S. (2016). Large-scale optimization algorithms for sparse conditional Gaussian graphical models. In Proceedings of the Conference on Artificial Intelligence and Statistics (pp. 528-537). PMLR.

McCarter, C., & Kim, S. (2014). On sparse Gaussian chain graph models. Advances in Neural Information Processing Systems, 27, 3212-3220.

The optimization method, Fast-CGGM and Mega-CGGM, in this package improve upon the previous optimization method by two to three orders of magnitude for the same statistical model described in the following papers:

Zhang, L., & Kim, S. (2014). Learning gene networks under SNP perturbations using eQTL datasets. PLoS computational biology, 10(2), e1003420.

Sohn, K. A., & Kim, S. (2012). Joint estimation of structured sparsity and output structure in multiple-output regression via inverse-covariance regularization. In Artificial Intelligence and Statistics (pp. 1081-1089). PMLR.

# Installation

## Linux

Run the following command to install the software.

```bash
./INSTALL.sh
```

## Mac OSX

Depending on the MacOSX version and default compiler, the location of some of the library #include statements in Mega-sCGGM and also those in Eigen for Fast-sCGGM need to change. Just always using the llvm/clang compiler installed via homebrew puts the libraries in the same place. For this, you will first need to install [Homebrew](https://brew.sh/), and then use it to install llvm:

```bash
brew install llvm
```

You will also need to make sure you have the latest versions of XCode command-line tools installed.

Then, run
```bash
./INSTALL.sh
```



# Running

See the [Python readme](README-Python.md) and [MATLAB readme](README-Matlab.md) to see how to use these interfaces.

To run from the command line, see the Demos directory:

```bash
cd Demos
./demo_simulated.sh
```
