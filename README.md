# PerturbNet
Software for Learning Gene Networks Underlying Clinical Phenotypes Using SNP Perturbations

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
