#REMINDER: INSERT SEEDS THING
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
from KSAT import KSAT
from SimAnn import simann
import pickle

# def probability( N, M_list, Rep = 1, n=30, mcmc_steps=8*N, anneal_steps=125,
#                 beta0=0.1, beta1=10.0, seeds=None):
#     Probl_list = []
#     Prob_for_N_array = [] #P_list_array
#     if seeds is None:
#         seeds = np.random.randint(10000000, size=len(M_list) * Rep)
#     for i in range(len(M_list)):
#         Prob_for_M_array = []
#         M = M_list[i]
#         used_seeds = []
#         for rep in range(Rep):
#             seed = int(seeds[rep + Rep * i])
#             np.random.seed(seed)
#             ksat = KSAT(N, M, 3)
#             solved = 0
#             for k in range(n):
#                 ksat.init_config()
#                 best = simann(ksat,
#                             anneal_steps=anneal_steps, mcmc_steps=mcmc_steps,
#                             beta0=beta0, beta1=beta1,
#                             seed=None,
#                             debug_delta_cost=False)
#                 if best.cost() == 0:
#                     solved += 1
#             P = solved / n
#             Prob_for_M_array.append(P)
#             used_seeds.append(seed)

#             #print(f"Percentage of problems solved with M = {M} is {P}")
#         Prob_for_N_array.append(Prob_for_M_array)
#         Probl_list.append(f"Ran trials for M = {M} with seed = {used_seeds}")
#     return  Prob_for_N_array, Probl_list
# Rep = 1
# N = 300
# M_list = [ 800, 900, 1000,1100,1200,1250,1300,1350]
# n_instances = 30
# mcmc_steps = 8 * N
# anneal_steps = 125
# beta0 = 0.1
# beta1 = np.log(mcmc_steps) + 1
# Prob_for_N_array, Probl_list = probability( N, M_list, Rep, n=n_instances,
#                     mcmc_steps=mcmc_steps,
#                     anneal_steps=anneal_steps,
#                     beta0=beta0,
#                     beta1=beta1)
# print(Prob_for_N_array)
# print(Probl_list)
# np_P_list_array = np.array(Prob_for_N_array)
# print(np_P_list_array)
# plt.figure(figsize=(8, 6))
# plt.plot(M_list, np.median(np_P_list_array, axis = 1), marker='o')
# plt.xlabel('Number of Clauses ')
# plt.ylabel(f' Probability P(N={N}, M)')
# plt.title('Probability of Solving 3-SAT with Simulated Annealing')
# plt.show()


#This function executes one trial of a simulated annealing algorithm 
#to solve a given 3-SAT instance.
#the input of the function is a tuple containing all the parameters

def single_iteration(argument):
    
    (N, M, seed, anneal_steps, mcmc_steps, beta0, beta1) = argument
    
    #If no seeds are provided, generate random seeds for reproducibility
    seed = np.random.randint(10000000)
    ksat = KSAT(N, M, 3, seed)
    #ksat.init_config()
    best = simann(ksat,
                    anneal_steps=anneal_steps, 
                    mcmc_steps=mcmc_steps,
                    beta0=beta0, 
                    beta1=beta1,
                    seed=None,
                    debug_delta_cost=False)
    #the output of the function returns 1 if the solution is found, 0 otherwise
    return 1 if best.cost() == 0 else 0



#The function evaluates how the success probability varies with 
#different values of 𝑁 (number of variables) and M (number of clauses).

#The function systematically increases 𝑀 by 50 
#and evaluates success probabilities for each value of N (N=[200,300,400,500,600])
#stopping if the problem becomes unsolvable two consecutive times (indicated by fails = 2).
#By recording the probabilities across multiple values of M for different N, 
#the function helps study how problem difficulty scales with N and M.
#Parallelization obtained thanks to multiprocessing.Pool speeds up the repeated simulations, 
#making the function efficient for large-scale experiments.


def probability(N_list, initial_M=400, Rep=5, n=30, mcmc_steps_factor=8, anneal_steps=125,
                beta0=0.1, beta1=10.0, seeds=None):
    
    #by default, it starts with M=400
    M = initial_M
    
    #this list is the final output which will be an array of arrays of arrays
    final_array = []
    
    #A list of log messages detailing the trials, including M values
    #Probl_list_array = []

    #the first larger loop iterates over all the possible values of N (N=[200,300,400,500,600])
    for N in N_list:
        
        mcmc_steps = N * mcmc_steps_factor  #the mcmc_steps change according to N (as N increases, they increase as well)
        
        beta1 = np.log(mcmc_steps) + 1   #beta1 is scaled according to N 

        #Probl_list = [] 
        Prob_for_N_array = []  #this array contains the list of probabilities for each N  

        fails = 0  #this is a counter for the number of failures
        while fails < 2:
            
            Prob_for_M_array = np.zeros(Rep, dtype = float)  #array of probability for each M initialized to all zeros 
            seed = None
            #Generate a list of arguments for parallel processing (`single_iteration`) 
            #using current values of N, M, annealing steps, etc.
            list_of_argument = [(N, M, seed, anneal_steps, mcmc_steps, beta0, beta1) for t in range(n)]
            for rep in range(Rep):
                #Perform parallel computation using a multiprocessing pool for each repetition.
                with mp.Pool() as pool:
                    res = pool.map(single_iteration, list_of_argument)
                print(res)
                #Calculate the probability P as the mean of the results from all trials.
                P = sum(res)/n
                #when the probability of success is zero, it means that none of the instances has been solved,
                #consequently we add 1 to the counter of the failures 
                if P == 0:
                    fails += 1
                if fails == 2:
                    
                    break

                Prob_for_M_array[rep] = P
            #Probl_list.append(f"Ran trials for M = {M}, with N = {N}") #INSERT LOG FOR SEED  
           
            #After completing Rep trials for a given M, increment `M` by 50
            M += 50
            Prob_for_N_array.append(Prob_for_M_array)
            #when two repetitions fail, then the loop breaks and it passes to the next N
            if fails == 2:
                break

        #A nested list where each entry corresponds to an N value, 
        #containing arrays of success probabilities for different M values.
        final_array.append(Prob_for_N_array)
        #Probl_list_array.append(Probl_list)


    return final_array, #Probl_list_array
def main():
   
    N_list = [200, 300, 400, 500, 600]
    try:
        result_prob  = probability( #♣test_array
            N_list, 
            start=400, 
            Rep=5, 
            n=32, 
            mcmc_steps_factor=8, 
            anneal_steps=125,
            beta0=0.1, 
            beta1=10.0, 
            seeds=None
        )
        print("Probability Results:")
        print(result_prob)
        #print("\nProblem Logs:")
        #print(test_array)

        # Save results using pickle
        with open("Array_1.pkl", "wb") as f:
            pickle.dump(result_prob, f)
        #with open("Array_2.pkl", "wb") as f:
            #pickle.dump(test_array, f)

    except Exception as e:
        print(f"An error occurred in main: {e}")
#if clause needed for the function to actually run and required by the multiprocessing module in python
#if __name__ == "__main__":
    #main()

