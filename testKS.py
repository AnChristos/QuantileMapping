import numpy as np
import scipy.stats
from QuantileMapping.NonParametricQMSpline import (
    fitNonParametricQMSpline)
from QuantileMapping.ParametricQM import (
    parametricQM)


def testKS():
    '''
    Create a high stat nominal and distored simul and derive correction.
    Then create smalll distorted samples correct then and plot the
    relevant KS distribution
    '''
    shift = 0.5
    smear = 1.2
    NumData = 30000
    NumSimul = 60000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)
    print('----->')
    print('KS of training data', scipy.stats.kstest(data, trueModel.cdf))
    print('KS of exact QM on training simul', scipy.stats.kstest(
        parametricQM(simul, trueModel, distortedModel),
        trueModel.cdf))
    # Do non-parametric QM correction
    LowPercentile = 0
    HighPercentile = 100
    numBins = 1000
    perc = np.linspace(
        LowPercentile, HighPercentile, numBins)
    QMnonParam = fitNonParametricQMSpline(data,
                                          simul,
                                          targetPerc=perc)

    print('KS of parametric qq QM on training', scipy.stats.kstest(
        QMnonParam.nominal(simul),
        trueModel.cdf))
    numForTest = 1000
    numExp = 1000
    KSpvalData = np.zeros(numExp)
    KSpvalExact = np.zeros(numExp)
    KSpvalQQ = np.zeros(numExp)
    for i in range(numExp):
        dataExp = trueModel.rvs(size=numForTest)
        simulExp = distortedModel.rvs(size=numForTest)
        _, KSpvalData[i] = scipy.stats.kstest(dataExp, trueModel.cdf)
        _, KSpvalExact[i] = scipy.stats.kstest(parametricQM(
            simulExp, trueModel, distortedModel), trueModel.cdf)
        _, KSpvalQQ[i] = scipy.stats.kstest(
            QMnonParam.nominal(simulExp), trueModel.cdf)

    for i in range(5, 20, 5):
        print('----->')
        print('KS excluding being from nominal at {: 2d} % level'
              '(using {} pseudo-experiments) '
              .format(i, numExp))
        perc = 100*len(KSpvalData[KSpvalData < 0.01*i])/numExp
        print('Generated from true model {:.0f} %'.format(perc))
        perc = 100*len(KSpvalExact[KSpvalExact < 0.01*i])/numExp
        print('Simulation corrected by QM exact {:.0f} %'.format(perc))
        perc = 100*len(KSpvalQQ[KSpvalQQ < 0.01*i])/numExp
        print('Simulation corrected by QM qq plot {:.0f} %'.format(perc))


if __name__ == "__main__":
    testKS()
