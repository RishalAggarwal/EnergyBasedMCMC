## Description

Code for the blog post [Models trained with unnormalized density functions: A need for a course correction](https://rishalaggarwal.github.io/ebmvsmcmc/)


This is a fork of the official repository for the paper [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://arxiv.org/abs/2402.06121). It is used primarily to setup MCMC experiments on the same systems IDEM was trained on. The environment is the same as the original [repository](https://github.com/jarridrb/DEM).

### Running MCMC

For GMM:

```python mcmc.py --energy gmm --n_chains 256000 --n_steps 100000 --step_size 1.25 --run_name gmm_final --save_endpoints True --endpoint_name gmm_endpoint1.npy```

For DW4:

```python mcmc.py --energy dw4 --n_chains 512000 --n_steps 100000 --step_size 0.5 --run_name dw4_old_baseline --box_size 2 --max_step 1 --save_endpoints True --endpoint_name dw4_endpoint1.npy```

For LJ13:

```python mcmc.py --energy lj13 --n_chains 512000 --n_steps 100000 --step_size 0.025 --run_name lj13_old_baseline --box_size 2.0 --max_step 0.1 --save_endpoints True --endpoint_name lj13_endpoint5.npy```

For LJ55:

```python mcmc.py --energy lj55 --n_chains 12800 --n_steps 200000 --step_size 0.0075 --run_name lj55_oldbaseline --box_size 2.0 --max_step 0.01 --num_equilibriation_steps 2000 --save_endpoints True --endpoint_name lj55_endpoint1.npy```

### Benchmarks

The benchmark evaluation codes are present in ```notebooks/<system>\_benchmark.ipynb``` files.

