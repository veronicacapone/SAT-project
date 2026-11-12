import numpy as np
import matplotlib.pyplot as plt

## Stochastically determine whether to accept a move according to the
## Metropolis rule (valid for symmetric proposals)
def accept(delta_c, beta):
    ## If the cost doesn't increase, we always accept
    if delta_c <= 0:
        return True
    ## If the cost increases and beta is infinite, we always reject
    if beta == np.inf:
        return False
    ## Otherwise the probability is going to be somewhere between 0 and 1
    p = np.exp(-beta * delta_c)
    ## Returns True with probability p
    return np.random.rand() < p

## The simulated annealing generic solver.
## Assumes that the proposals are symmetric.
## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move - must be symmetric!
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent object
## NOTE: The default beta0 and beta1 are arbitrary.
def simann(probl,
           anneal_steps=10, mcmc_steps=100,
           beta0=0.1, beta1=10.0,
           seed=None, debug_delta_cost=False):
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)

    # Set up the list of betas.
    # First allocate an array with the required number of steps
    beta_list = np.zeros(anneal_steps)
    # All but the last one are evenly spaced between beta0 and beta1 (included)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    # The last one is set to infinity
    beta_list[-1] = np.inf

    
    probl.init_config() 
    c = probl.cost()
    print(f"initial cost = {c}")
    # print(probl.x)

    ## Keep the best cost seen so far, and its associated configuration.
    best = probl.copy()
    best_c = c

    acceptance_rates = []
    delta_costs = [] 

    # Signal if we have a solution
    solution_found = False

    # Main loop of the annealing: Loop over the betas
    for beta in beta_list:
        ## We want to record the acceptance rate for each beta, so we need a
        ## counter "accepted" for the number of accepted moves
        accepted = 0
        # For each beta, perform a number of MCMC steps
        for t in range(mcmc_steps):
            move = probl.propose_move()
            # print(probl.x)
            delta_c = probl.compute_delta_cost(move)
            ## Optional (expensive) check that `compute_delta_cost` works
            if debug_delta_cost:
                probl_copy = probl.copy()
                probl_copy.accept_move(move)
                assert abs(c + delta_c - probl_copy.cost()) < 1e-10
            ## Metropolis rule
            # print(probl.x, c, move, delta_c, accept(delta_c, beta))

            # Collect delta_c values
            delta_costs.append(delta_c)

            if accept(delta_c, beta):
                probl.accept_move(move)
                # print(probl.x)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()
                    if best_c == 0:
                        solution_found = True
                        break
        print(f"acc.rate={accepted / mcmc_steps} beta={beta} c={c} [best={best_c}]")

        acceptance_rates.append(accepted / mcmc_steps)

        if solution_found:
            break

# Plot for the acceptance rate evolution
    plt.figure(figsize=(8,6))
    plt.yticks(ticks = np.arange(0,1.1,0.1))
    plt.plot(beta_list[:len(acceptance_rates)], acceptance_rates, label='Acceptance Rate')
    plt.plot(beta_list[:len(acceptance_rates)], np.exp(-beta_list[:len(acceptance_rates)]))
    plt.xlabel('Beta')
    plt.ylabel('Acceptance Rate')
    plt.title('Acceptance Rate Evolution for M = M')
    plt.legend()
    plt.show()

# Plot of the histogram of delta costs
    plt.figure()
    plt.hist(delta_costs, bins=50, edgecolor='k')
    plt.xlabel('Delta Cost')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delta Costs with M = M')
    plt.show()


    ## Return the best instance
    print(f"final cost = {best_c}")
    return best