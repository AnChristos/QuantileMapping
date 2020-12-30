import numpy as np
import scipy.stats
import scipy.interpolate
from QuantileMapping.NonParametricQMSpline import (
    fitNonParametricQMSpline)
from QuantileMapping.ParametricQM import (
    parametricQM)
from CommonPlottingHelper import (
    compareHist, compareCDF, compareMethods, compareCorrection)


def testExample():
    '''
    Create a nominal and a distorted distribution

    Apply:
     - exact parametric QM since we know the exact true model
     - non parametric QM pretending we do not know the true model
    '''
    shift = 0.5
    smear = 1.2
    NumData = 40000
    NumSimul = 120000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)

    # Do pefect parametric correction
    exactQMCorr = parametricQM(simul, trueModel, distortedModel)

    # Do non-parametric QM correction
    LowPercentile = 0.1
    HighPercentile = 99.9
    numBins = 1000
    perc = np.linspace(
        LowPercentile, HighPercentile, numBins)
    QMqq = fitNonParametricQMSpline(data,
                                    simul,
                                    targetPerc=perc,
                                    numBootstrap=2000)

    # Compare the corrections derived into certain points
    compareCorrection(points=QMqq.uncorrected,
                      QMExact=parametricQM(
                          QMqq.uncorrected, trueModel, distortedModel),
                      NonParametricQQ=QMqq.nominal,
                      title="Compare corrections",
                      name="CorrectionCompare.png")

    # interpolate the qq correction not
    interQMCorr = scipy.interpolate.interp1d(
        QMqq.uncorrected,
        QMqq.nominal,
        fill_value='extrapolate',
        assume_sorted=True)
    nonParamQMCorr = interQMCorr(simul)
    # pdf histograms
    # window for histograms
    minhist = -5
    maxhist = 5
    histBins = 50
    cdfBins = 100
    binning = np.linspace(minhist, maxhist, histBins)
    cdfbinning = np.linspace(minhist, maxhist, cdfBins)

    compareMethods(data=data,
                   simul=simul,
                   QMExact=exactQMCorr,
                   NonParametricQQ=nonParamQMCorr,
                   binning=binning,
                   title="Compare Methods",
                   name="MethodCompare.png")

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
               cdfbinning=cdfbinning,
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
               cdfbinning=cdfbinning,
               title='Non - Parametric QM',
               name='NonParamQMcdf.png')


if __name__ == "__main__":
    testExample()
