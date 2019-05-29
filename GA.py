# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
"""

import numpy as np
import matplotlib.pyplot as plt


def GA(n, max_evals, decodefct, selectfct, fitnessfct, min_attainable=0, seed=None):
    eval_cntr = 0
    history = []
    #
    # GA params
    mu = 100
    # pc = 0.5
    pm = 1
    iter = 0

    local_state = np.random.RandomState(seed)
    Genome = np.zeros((mu, n), int)

    for q in range(mu):
        perm = local_state.permutation(n)+1
        Genome[q] = np.copy(perm)
    Phenotype = np.zeros((mu, n), int)
    for k in range(mu):
        Phenotype[k] = np.copy(decodefct(Genome[k]))

    fitness = fitnessfct(Phenotype)
    eval_cntr += mu
    fcurr_best = fmin = np.min(fitness)
    xcurr_best = xmin = Genome[np.argmin(fitness)]
    history.append(fmin)
    while eval_cntr < max_evals:
        iter += 1
        # Generate offspring population (recombination, mutation)
        newGenome = np.empty([mu, n], dtype=int)
        # 1. sexual selection + 2-point recombination
        for i in range(int(mu/2)):
            x1 = local_state.randint(n)
            x2 = local_state.randint(n)

            parent1 = Genome[x1]
            parent2 = Genome[x2]


            if 0:  # local_state.uniform() < 1:  # recombination
                idx = 0
                while idx == 0:
                    idx = local_state.randint(n, dtype=int)

                Xnew1 = np.copy(perm_crossover(parent1, parent2, local_state.randint(n), local_state.randint(n)))
                Xnew2 = np.copy(perm_crossover(parent2, parent1, local_state.randint(n), local_state.randint(n)))
            # else:  # no recombination; two parents are copied as are
                # Xnew1 = np.copy(p1)
                # Xnew2 = np.copy(p2)
#        2. mutation
            if local_state.uniform() < pm:
                # ind_1 = local_state.randint(n)
                # ind_2 = local_state.randint(n)
                # Xnew1[ind_1], Xnew1[ind_2] = Xnew1[ind_2], Xnew1[ind_1]
                Xnew1 = np.copy(variation(parent1))

            if local_state.uniform() < pm:
                # ind_1 = local_state.randint(n)
                # ind_2 = local_state.randint(n)
                # Xnew2[ind_1], Xnew2[ind_2] = Xnew2[ind_2], Xnew2[ind_1]
                Xnew2 = np.copy(variation(parent2))
#
            newGenome[2*i-1] = np.copy(Xnew1)
            newGenome[2*i] = np.copy(Xnew2)
        # The best individual of the parental population is kept
        wild_card = np.copy(Genome[local_state.randint(n)])
        it_num = local_state.randint(n)
        for it in range(it_num):
            cur_xover = Genome[local_state.randint(n)]
            new = perm_crossover(wild_card, cur_xover, local_state.randint(n), local_state.randint(n))
            np.append(newGenome, new)
        newGenome[[mu-1], :] = np.copy(Genome[[np.argmin(fitness)], :])
        Genome = np.copy(newGenome)

        for k in range(mu):
            Phenotype[k] = np.copy(decodefct(Genome[k]))
        fitness = fitnessfct(Phenotype)
        eval_cntr += mu
        fcurr_best = np.min(fitness)
        xcurr_best = Genome[np.argmin(fitness)]
        if fmin < fcurr_best:
            fmin = fcurr_best
            xmin = xcurr_best

        history.append(fcurr_best)
        if np.mod(eval_cntr, int(max_evals/10)) == 0:
            print("iter:", iter, " evals:", eval_cntr, " fmin=", fmin, " xmin= ", xmin)
        if fmin == min_attainable:
            print(eval_cntr, " evals: fmin=", fmin, "; done!")
            break
    return xmin, fmin, history


def perm_crossover(parent1, parent2, indx1, indx2):  # MORT this works!
    p1 = np.copy(parent1)
    p2 = np.copy(parent2)
    n = len(p1)
    ret = np.zeros(n, int)
    left_index = min(indx1, indx2)
    right_index = max(indx1, indx2)

    child_chromosome = np.zeros(right_index-left_index, int)
    for l in range(left_index, right_index):
        child_chromosome[l-left_index] = p1[l]

    for a in range(0, len(child_chromosome)):
        temp = np.delete(p2, np.where(p2 == child_chromosome[a]))
        p2 = np.copy(temp)

    counter = 0
    for m in range(0, len(ret)):
        if left_index <= m < right_index:
            ret[m] = child_chromosome[m-left_index]
        else:
            ret[m] = p2[counter]
            counter += 1

    # for m in range(len(child_chromosome)):
    #     itemindex = np.where(p2 == child_chromosome[m])
    #     p2 = np.delete(p2, itemindex)
    #
    # print("p2: ", p2)
    #
    # remember_index = 0
    # while remember_index <= len(child_chromosome):
    #     ret[remember_index] = child_chromosome[remember_index]
    #     remember_index += 1
    #
    # for j in range(0, len(p2)):
    #     print(j)
    #     ret[remember_index+j] = p2[j]
    #
    return ret



def no_decoding(a):
    ''' The identity function when no decoding is needed, e.g., search over binary landscapes. '''
    ret = np.copy(a)
    return ret


def decoding_ones(a):
    z = a < 1
    a[z] = -1
    return a


def fitness(gen):  # MORT: this works!
    fitness_vect = np.zeros(len(gen))
    for s in range(len(gen)):
        fitness_vect[s] = fitness_perm(gen[s])

    return fitness_vect


def fitness_perm(chromosome):  # MORT this works!
    n = len(chromosome)
    counter_l = 0
    counter_r = 0
    for u in range(1, n):
        cur_val = chromosome[u-1]
        left_i = (u-1)-1
        while left_i >= 0:
            left_val = chromosome[left_i]
            if abs((u-1)-left_i) == abs(left_val-cur_val):
                counter_l += 1
            left_i -= 1
        right_i = (u-1)+1
        while right_i < n:
            right_val = chromosome[right_i]
            if abs(right_i-(u-1)) == abs(right_val-cur_val):
                counter_r += 1
            right_i += 1
    return counter_l+counter_r


def select_proportional(Genome, fitness, rand_state):
    cumsum_f = np.cumsum(fitness)
    r = sum(fitness) * rand_state.uniform()
    idx = np.ravel(np.where(r < cumsum_f))[rand_state.randint(len(np.where(r < cumsum_f)))]
    ret = Genome[idx]
    return ret


    # index1 = rand_state.randint(n)
    # index2 = rand_state.randint(n)
    # index3 = rand_state.randint(n)
    # [fit1,fit2, fit3] = [fitness[index1], fitness[index2], fitness[index3]]
    # MORT
    # best_fit , best_i = max()


def insert(xx):
    x = np.copy(xx)
    n = len(x)
    move = np.random.randint(0, n-1)
    index = np.random.randint(0, n-1)
    temp = np.delete(x, move)
    to_ret = np.insert(temp, index, x[move])
    return to_ret


def swap(xx):
    x = np.copy(xx)
    num_swaps = np.random.randint(1, 3)
    for _ in range(np.int(num_swaps)):
        p = np.random.randint(0, len(xx) - 1)
        p2 = np.mod((p+1), len(xx))
        tmp = x[p]
        x[p] = x[p2]
        x[p2] = tmp
    return x


def inverse(xx):
    arr = xx.tolist()
    p1 = np.random.randint(0, len(arr) - 1)
    p2 = np.random.randint(0, len(arr) - 1)
    a = min(p1, p2)
    b = max(p1, p2)
    to_ret = arr[:a] + arr[a:b][::-1] + arr[b:]
    return np.array(to_ret)


def variation(xx):
    x1 = swap(xx)
    x2 = inverse(xx)
    x3 = insert(xx)

    f1 = fitness_perm(x1)
    f2 = fitness_perm(x2)
    f3 = fitness_perm(x3)

    fmin = min(f1, f2, f3)
    if fmin == f1:
        return x1
    elif fmin == f2:
        return x2
    else:
        return x3



def threeWayTournament(par1, par2):  # MORT: this works!
    n = len(par1)
    # ret_child = [0] * n
    # perm_set_cur = list(range(1, n+1))
    #
    # for a in range(1, n):
    #     if par1[a-1] == par2[a-1]:
    #         ret_child[a-1] = par1[a-1]
    #         perm_set_cur.remove(par1[a-1])
    # for b in range(len(ret_child)):
    #     if ret_child[b] == 0:
    #         rnd_ind = np.random.randint(len(perm_set_cur))
    #         ret_child[b] = perm_set_cur[rnd_ind]
    #         perm_set_cur.remove(ret_child[b])
    # return ret_child
    par1mutation = np.random.uniform(0, 1) < 0.5
    if par1mutation:
        child = np.copy(par1)
    else:
        child = np.copy(par2)
    ind_1 = np.random.randint(n)
    ind_2 = np.random.randint(n)
    child[ind_1], child[ind_2] = child[ind_2], child[ind_1]

    return child


if __name__ == "__main__":
    n = 16
    evals = 10 ** 6
    Nruns = 3
    fbest = []
    xbest = []
    for d in range(Nruns):
        xmin, fmin, history = GA(n, evals, no_decoding, select_proportional, fitness, 0, np.random.randint(10**5))
        plt.semilogy(np.array(history))
        plt.show()
        print(d, ": maximal N-Queens found is ", n-fmin, " Permutation is: ", xmin)
        fbest.append(fmin)
        xbest.append(xmin)
    print("====\n Best ever: ", min(fbest), "x*=", xbest)
    # xx1 = [9, 8, 6, 2, 5, 7, 1, 4, 3, 10]
    # fit = fitness_perm(xx)
    # print("fitness: ", fit)
    # xx2 = [6, 8, 1, 4, 5, 7, 2, 10, 3, 9]
    # xx2 = [3, 1, 4, 2]
    # xx3 = [4, 1, 5, 8, 2, 7, 3, 6]
    # xx3 = [10, 8, 6, 2, 9, 11, 1, 5, 7, 3, 4]
    # fit = fitness_perm(xx)
    # fit8 = fitness_perm(xx2)
    # fit11 = fitness_perm(xx3)
    # p = threeWayTournament(xx1,xx2)
    # i1 = np.random.randint(n)
    # i2 = np.random.randint(n)
    # p = perm_crossover(xx1, xx2, i1, i2)
    # print("i1:", i1, " i2:", i2)
    # print("p: = ", p)
    # print("fitness10= ", fit, " fitness8=", fit8, " fitness11= ", fit11)
