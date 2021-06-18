# QuantileMapping

Playground for testing Quantile Mapping 


function ParametricQM 
----------------------------------------------------
Module : QuantileMapping.ParametricQM

Apply exact formula  F^1_data F_simul(x) 

correction.ppf(simulation.cdf(input))

When the ppf and cdf are known this is exact

class QMqqMap
-----------------------------
Module : QuantileMapping.QMqqMap

Create a quantile - quantile map,
including uncertainties

function npCI
--------------------
Module : QuantileMapping.QuantileHelper

Implement approximate non parametric confidence intervals
for quantiles.

Following
"Confidence interval for quantiles and percentiles",

Cristiano Ialongo

doi: 10.11613/BM.2019.010101


Example 1 : testExampleFit
------------------------------

Introduce distortion to a normal distibution.

Derive a correction by fitting the quantile-quantile plot

Compare it with the known exact Quantile Map correction 

Example 2 : testExampleInterpolation
------------------------------

Introduce distortion to a normal distibution.

Derive a correction by interpolating the quantile-quantile plot

Compare it with the known exact Quantile Map correction.


