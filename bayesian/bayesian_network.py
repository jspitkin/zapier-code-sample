# Jake Pitkin
# Feb 19 2018
#
import pandas as pd
import numpy as np
import copy
from functools import reduce
pd.options.mode.chained_assignment = None

def productFactor(A, B):
    """ Computes the factor between A and B and returns the result.
        Assumes the product of A and B is a valid operation.
    """
    C = reshape(A, B)
    D = reshape(B, A)
    for index, row in C.iterrows():
        C.loc[index, 'probs'] = C.loc[index, 'probs'] * D.loc[index, 'probs']
    return C

def marginalizeFactor(A, margVar):
    """ Marginalizes margVar from a single factor A and returns the result.
        Assume that margVar appears on the left side of the conditional.
    """
    new_vars = A.axes[1][1:]
    new_vars = new_vars.drop(margVar)
    level_list = []
    rows = 1
    for var in new_vars:
        level_list.append(A[var].unique())
        rows *= len(A[var].unique())
    probabilities = [0] * rows
    B = createCPT(new_vars, probabilities, level_list)
    for index, row in B.iterrows():
        rel_values = []
        for var in new_vars:
            rel_values.append(B.loc[index, var])
        B.loc[index, 'probs'] = get_prob_sum(A, new_vars, rel_values)
    return B

def marginalize(bayesNet, margVars):
    """ Takes in a bayesNet and marginalizes out all variables in margVars
        and returns the result. This is done using variable elimination.
    """
    marg_net = copy.deepcopy(bayesNet)
    ordered_vars = minimum_degree(marg_net, margVars)
    ordered_vars = ['b', 'e']
    for var in ordered_vars:
        tables_to_factor = []    
        tables_to_keep = []
        for table in marg_net:
            if var in table.axes[1][1:]:
                tables_to_factor.append(table)
            else:
                tables_to_keep.append(table)
        if len(tables_to_factor) != 0:
            factor = factor_tables(tables_to_factor)
            marg_factor = marginalizeFactor(factor, var)
            marg_net = tables_to_keep + [marg_factor]
        else:
            marg_net = tables_to_keep 
    return marg_net

def observe(bayesNet, obsVars, obsVals):
    """ Takes in a bayesNet and and sets the list of variables obsVars
        to the corresponding list of values obsVals and returns the result. 
        The factors are not normalized as probabilities.
    """
    observe_net = copy.deepcopy(bayesNet)
    for table in observe_net:
        table_vars = table.axes[1][1:]
        for index, row in table.iterrows():
            for var_index, var in enumerate(obsVars):
                if var not in table_vars:
                    continue
                if table.loc[index, var] != obsVals[var_index]:
                    table.drop(index, inplace=True)
                    break
    return observe_net

def infer(bayesNet, margVars, obsVars, obsVals):
    """ Takes in a bayesNet and returns a single joint proability table
        resulting from observing a set of variables and marginalizing a
        set of variables. The values in the table are normalized as
        probabilities.
    """
    observed_net = observe(bayesNet, obsVars, obsVals)
    marg_net = marginalize(observed_net, margVars)
    print(marg_net[0])
    print(marg_net[1])
    print(marg_net[2])
    factor = factor_tables(marg_net)
    print(factor)
    return normalize(factor)

def createCPT(varnames, probs, levelsList):
    """ Constructs a conditional probability table.
        Written by Tom Fletcher. 
    """
    cpt = pd.DataFrame({'probs': probs})

    m = len(probs)
    n = len(varnames)

    k = 1
    for i in range(n - 1, -1, -1):
        levs = levelsList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        cpt[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    return cpt

def createCPTfromData(data, varnames):
    """ Originally written by Tom Fletcher. 
        With some optimizations by Maks Cegielski-Johnson.
    """
    numVars = len(varnames)
    levelsList = []

    for i in range(0, numVars):
        name = varnames[i]
        levelsList = levelsList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), levelsList))
    m = reduce(lambda x, y: x * y, lengths)
    n = len(varnames)

    cpt = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(n - 1, -1, -1):
        levs = levelsList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        cpt[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(levelsList[0])
    skip = int(m / numLevels)

    ## This chunk of code creates the vector "fact" to index into probs using
    ## matrix multiplication with the data frame data
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(levelsList[i])

    ## Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    temp_counter = dict((person,0) for person in range(1,m+1))
    for ap in a:
        temp_counter[ap] += 1
    for i in range(0, m):
        cpt.set_value(i,"probs", temp_counter[i+1])


    # Now normalize the conditional probabilities
    for i in range(0, skip):
        denom = 0
        for j in range(i, m, skip):
            denom = denom + cpt['probs'][j]
        for j in range(i, m, skip):
            if denom != 0:
                cpt.set_value(j, "probs",cpt['probs'][j] / denom)

    return cpt

def normalize(factor):
    """ Given a single table, returns a normalized version. """
    Z = sum(factor['probs'])
    norm_factor = factor[:]
    for index, rows in norm_factor.iterrows():
        normalized = norm_factor.loc[index, 'probs'] / Z
        norm_factor.loc[index, 'probs'] = normalized
    return norm_factor

def factor_tables(tables):
    """ Takes a list of tables and returns a single factor table. """
    if len(tables) == 0:
        return None
    if len(tables) == 1:
        return tables[0]
    factor = productFactor(tables[0], tables[1])
    for table in tables[2:]:
        factor = productFactor(factor, table)
    return factor

def get_prob(A, var_list, value_list):
    """ Returns the probability from a table A where all values in var_list
        are set to their respective values in value_list. -1 if such a 
        row doesn't exist.
    """
    for A_index, row in A.iterrows():
        match = True
        for index, var in enumerate(var_list):
            if A.loc[A_index, var] != value_list[index]:
                match = False
                break
        if match:
            return A.loc[A_index, 'probs']
    return -1

def get_prob_sum(A, var_list, value_list):
    """ Returns the sum of probabilities from a table A where all the values
        in var_list are set to their respective values in value_list. -1 if no
        such rows exist.
    """
    sum = 0
    for A_index, row in A.iterrows():
        match = True
        for index, var in enumerate(var_list):
            if A.loc[A_index, var] != value_list[index]:
                match = False
                break
        if match:
            sum += A.loc[A_index, 'probs']
    return sum

def reshape(A, B):
    """ Returns of probability table C that is of the proper shape to
        represent the product of A and B. The probabilities in the table
        will only reflect the variables in table A.
    """
    A_vars = A.axes[1][1:]
    B_vars = B.axes[1][1:]
    new_vars = A_vars.union(B_vars).unique()
    level_list = []
    rows = 1
    for var in new_vars:
        if var in A_vars:
            level_list.append(A[var].unique())
            rows *= len(A[var].unique())
        else:
            level_list.append(B[var].unique())
            rows *= len(B[var].unique())
    probabilities = [0] * rows
    C  = createCPT(new_vars, probabilities, level_list)
    for index, row in C.iterrows():
        rel_values = []
        for var in A_vars:
            rel_values.append(C.loc[index, var])
        C.loc[index, 'probs'] = get_prob(A, A_vars, rel_values)
    return C

def minimum_degree(bayesNet, variables):
    """ Takes in a Bayesian network and a collection of variables.
        Returns the variable that will result in the smallest factor. 
        Used as a heuristic for variable elimination order in the
        variable elimination algorithm.
    """
    var_counts = {}
    for var in variables:
        counted_vars = set()
        count = 1
        for table in bayesNet:
            if var not in table.axes[1][1:]:
                continue
            for table_var in table.axes[1][1:]:
                if table_var not in counted_vars:
                    count *= len(table[table_var].unique())
                    counted_vars.add(table_var)
        var_counts[var] = count
    return sorted(var_counts, key=lambda k: var_counts[k])
