# Empirical-Bayesian-Learning-in-AR-Graphical-Models

Code title: "Empirical-Bayesian-Learning-in-AR-Graphical-Models"

Implementation of the ADMM algorithm (c) 2015-2016 Raphaël Liégeois R.Liegeois@ulg.ac.be and Bamdev Mishra B.Mishra@ulg.ac.be.

This package contains a MATLAB implementation of the algorithm proposed in the report.

Raphaël Liégeois, Bamdev Mishra, Mattia Zorzi, and Rodolphe Sepulchre, "Sparse plus low-rank autoregressive identification in neuroimaging time series", Technical report, arXiv:1503.08639, 2015.

This implementation is due to Raphaël Liégeois R.Liegeois@ulg.ac.be and Bamdev Mishra B.Mishra@ulg.ac.be, 2015.

The implementation is a research prototype still in development and is provided AS IS. No warranties or guarantees of any kind are given. Do not distribute this code or use it other than for your own research without permission of the authors.

Feedback is greatly appreciated.

Installation:
Run "Install_mex.m". You do not need to do this step for subsequent usage.
Run "Run_me_first.m" to add folders to the working path. This needs to be done at the starting of each session.
To check that everything works, run "Test.m" at Matlab command prompt (you should see some plots at the end).
Files:
ADMM_sparse_lowrank_AR.m: This is the main file related to the technical report mentioned above. It also contains 16 functions that are within the .m file.
projectSortC.c: This c file corresponds to the projection onto the L1 ball.
Test.m: A test file.
Examples: This folder contains some examples.
