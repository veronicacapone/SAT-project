import SimAnn
import KSAT
import numpy as np

## Generate a problem to solve.
N = 200
#M = [200, 400, 600, 800, 1000]
mcmc_steps = 8*N
anneal_steps = 125
beta0 = 0.1
beta1 = np.log(mcmc_steps) + 1

ksat = KSAT.KSAT(N = 200, M = 200, K =3, seed = 42)

## Optimize it.
best = SimAnn.simann(ksat,
                     mcmc_steps, anneal_steps,
                     beta0, beta1, seed = 42,
                     debug_delta_cost = False)
