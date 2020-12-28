import numpy as np
import scipy.stats
from QuantileMapping.NonParametricQMSpline import (
    fitNonParametricQMSpline)
from QuantileMapping.ParametricQM import (
    parametricQM)
from CommonPlottingHelper import (
    compareHist, compareCDF)


def testlogNormal():
    '''
    Create a nominal and a distorted invert normal
    distribution.

    Apply:
     - exact parametric QM since we know the exact true model
     - non parametric QM pretending we do know the true model
    '''
    distortion = 0.2
    lognormS = 0.5
    NumData = 10000
    NumSimul = 50000
    trueModel = scipy.stats.lognorm(s=lognormS)
    distortedModel = scipy.stats.lognorm(s=lognormS, loc=distortion)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)

    # Do pefect parametric correction
    exactQMCorr = parametricQM(simul, trueModel, distortedModel)

    # Do non-parametric QM correction
    LowPercentile = 0.0
    HighPercentile = 100
    numBins = 1000
    perc = np.linspace(
        LowPercentile, HighPercentile, numBins)
    QMnonParam = fitNonParametricQMSpline(data,
                                          simul,
                                          targetPerc=perc,
                                          bootstrapMode='data')

    nonParamQMCorr = QMnonParam.nominal(simul)

    # pdf histograms
    # window for histograms
    minhist = 0
    maxhist = 5
    histBins = 75
    binning = np.linspace(minhist, maxhist, histBins)
    compareHist(data=data,
                simul=simul,
                corrected=exactQMCorr,
                trueModel=trueModel,
                binning=binning,
                title='Using "exact" Parametric QM',
                name='ExactQMpdf.png')

    compareCDF(data=data,
               simul=simul,
               corrected=exactQMCorr,
               trueModel=trueModel,
               cdfbinning=binning,
               title='Using "exact" Parametric QM',
               name='ExactQMcdf.png')

    compareHist(data=data,
                simul=simul,
                corrected=nonParamQMCorr,
                trueModel=trueModel,
                binning=binning,
                title='Non - Parametric QM',
                name='NonParamQMpdf.png')

    compareCDF(data=data,
               simul=simul,
               corrected=nonParamQMCorr,
               trueModel=trueModel,
               cdfbinning=binning,
               title='Non - Parametric QM',
               name='NonParamQMcdf.png')


if __name__ == "__main__":
    testlogNormal()
