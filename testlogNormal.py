import numpy as np
import scipy.stats
from QuantileMapping.NonParametricQMSpline import (
    fitNonParametricQMSpline)
from QuantileMapping.ParametricQM import (
    parametricQM)
from CommonPlottingHelper import (
    compareHist)


class logNormWrapper:
    def __init__(self, s, loc):
        self.s = s
        self.loc = loc

    def cdf(self, x):
        return scipy.stats.lognorm.cdf(x, self.s, self.loc)

    def ppf(self, q):
        return scipy.stats.lognorm.ppf(q, self.s, self.loc)


def testlogNormal():
    '''
    Create a nominal and a distorted invert normal
    distribution.

    Apply:
     - exact paramtric QM
     - non parametric QM
    '''
    distortion = 0.25
    lognormS = 0.5
    NumData = 10000
    NumSimul = 50000
    data = scipy.stats.lognorm.rvs(s=lognormS,
                                   loc=0,
                                   size=NumData)
    simul = scipy.stats.lognorm.rvs(s=lognormS,
                                    loc=distortion,
                                    size=NumSimul)

    # window for histograms
    minhist = 0
    maxhist = 5
    histBins = 75

    # Do pefect parametric correction
    knownppf = logNormWrapper(s=lognormS, loc=0)
    knowncdf = logNormWrapper(s=lognormS, loc=distortion)
    exactQMCorr = parametricQM(simul, knownppf, knowncdf)
    compareHist(data=data,
                simul=simul,
                corrected=exactQMCorr,
                minhist=minhist,
                maxhist=maxhist,
                histBins=histBins,
                title='Using "exact" Parametric QM',
                name='ExactQM.png')

    # Do non-parametric QM correction
    LowPercentile = 0.0
    HighPercentile = 100
    numBins = 500
    perc = np.linspace(
        LowPercentile, HighPercentile, numBins)
    QMnonParam = fitNonParametricQMSpline(data,
                                          simul,
                                          targetPerc=perc,
                                          bootstrapMode='data')

    nonParamQMCorr = QMnonParam.nominal(simul)
    compareHist(data=data,
                simul=simul,
                corrected=nonParamQMCorr,
                minhist=minhist,
                maxhist=maxhist,
                histBins=histBins,
                title='Non - Parametric QM',
                name='NonParamQM.png')


if __name__ == "__main__":
    testlogNormal()
