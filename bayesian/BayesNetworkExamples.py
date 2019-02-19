from bayesian_network import *

####################################
## Simple chain example: x -> y -> z
####################################
x = createCPT(['x'], [0.3, 0.7], [['T','F']])
yx = createCPT(['y','x'], [0.8, 0.4, 0.2, 0.6], [['T','F'], ['T','F']])
zy = createCPT(['z','y'], [0.5, 0.6, 0.5, 0.4], [['T','F'], ['T','F']])

xyzNet = [x, yx, zy]

## Some simple operations you might try to check your code
productFactor(x, yx)
productFactor(productFactor(x, yx), zy)
marginalizeFactor(productFactor(x, yx), 'x')
marginalizeFactor(productFactor(yx, zy), 'z')

## Notice in the observe function, you just need to delete rows that are
## inconsistent with the given observations. Factors do not need to be combined
## or normalized in this step.
observe(xyzNet, 'x', 'T')
observe(xyzNet, ['x', 'y'], ['T', 'T'])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
marginalize(xyzNet, 'x')
marginalize(xyzNet, 'y')
marginalize(xyzNet, 'z')
marginalize(xyzNet, ['x', 'z'])

#############################
## Bishop book (Ch 8) example
#############################
b = createCPT(['battery'], [0.9, 0.1], [[1, 0]])
f = createCPT(['fuel'], [0.9, 0.1], [[1, 0]])
gbf = createCPT(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [b, f, gbf]

## Some examples:
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
productFactor(productFactor(b, f), gbf)
productFactor(productFactor(gbf, f), b)

marginalizeFactor(productFactor(gbf, b), 'gauge')
productFactor(marginalizeFactor(gbf, 'gauge'), b)

productFactor(marginalizeFactor(productFactor(gbf, b), 'battery'), f)
marginalizeFactor(productFactor(productFactor(gbf, f), b), 'battery')

marginalizeFactor(productFactor(marginalizeFactor(productFactor(gbf, b), 'battery'), f), 'gauge')
marginalizeFactor(productFactor(marginalizeFactor(productFactor(gbf, b), 'battery'), f), 'fuel')

## Examples computed in book (see pg. 377)
A = infer(carNet, ['battery', 'fuel'], [], [])      ## (8.30)
print("Expected: {} Actual: {}".format(0.315, A.iloc[1]['probs']))
B = infer(carNet, ['battery'], ['fuel'], [0])           ## (8.31)
print("Expected: {} Actual: {}".format(0.81, B.iloc[1]['probs']))
C = infer(carNet, ['battery'], ['gauge'], [0])          ## (8.32)
print("Expected: {} Actual: {}".format(0.257, C.iloc[1]['probs']))
D = infer(carNet, [], ['gauge', 'battery'], [0, 0]) ## (8.33)
print("Expected: {} Actual: {}".format(0.111, D.iloc[1]['probs']))

###########################################################################
## Kevin Murphy's Example: http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
###########################################################################
c = createCPT(['cloudy'], [0.5, 0.5], [ ['F', 'T'] ])
rc = createCPT(['rain', 'cloudy'], [0.8, 0.2, 0.2, 0.8], [ ['F', 'T'], ['F', 'T'] ])
sc = createCPT(['sprinkler', 'cloudy'], [0.5, 0.9, 0.5, 0.1], [ ['F', 'T'], ['F', 'T'] ])
wsr = createCPT(['wet', 'sprinkler', 'rain'], [1, 0.1, 0.1, 0.01, 0, 0.9, 0.9, 0.99], [ ['F', 'T'], ['F', 'T'], ['F', 'T'] ])

grassNet = [c, rc, sc, wsr]

## Test your infer() method by replicating the computations on the website!!
