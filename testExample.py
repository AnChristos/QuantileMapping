import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
from QuantileMapping.QMqqMap import (
    QMqqMap)
from QuantileMapping.ParametricQM import (
    parametricQM)


def testExampleInterpolation():
    '''
    Create a nominal and a distorted distribution
    Study :
     - exact parametric QM since we know the exact true model
     - non parametric QM pretending we do not know the true model
    '''
    shift = 0.5
    smear = 1.2
    NumData = 10000
    NumSimul = 40000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)

    # Do non-parametric QM correction
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=1.,
        endPerc=99.,
        numPoints=100,
        sigma=1.96)

    QMqq.savetxt("qqPlot.txt")
    Xminus = QMqq.X - QMqq.Xlow
    Xplus = QMqq.Xup - QMqq.X
    errorX = np.row_stack((Xminus, Xplus))

    Yminus = QMqq.Y - QMqq.Ylow
    Yplus = QMqq.Yup - QMqq.Y
    errorY = np.row_stack((Yminus, Yplus))

    # Fix colours for plotting
    dataColour = 'black'
    exactColour = 'skyblue'
    approxColour = 'red'
    simulColour = 'forestgreen'

    # Compare estimated with exact QM
    fig, ax = plt.subplots()
    ax.errorbar(x=QMqq.X,
                y=QMqq.Y,
                xerr=errorX,
                yerr=errorY,
                marker='.',
                ls='none',
                markersize=2,
                color=approxColour,
                label='Estimated QM')
    ax.plot(QMqq.X,
            parametricQM(QMqq.X, trueModel, distortedModel),
            color=exactColour,
            label='Exact QM')
    ax.legend(loc='best')
    ax.set(xlabel='Input ', ylabel='Corrected input')
    ax.set_title('Estimated vs Perfect QM Correction')
    fig.savefig('CorrectionCompare.png', dpi=300)

    # Do pefect parametric correction
    exactQMCorr = parametricQM(simul, trueModel, distortedModel)
    # Use interpolation for  the qq correction
    interQMCorr = scipy.interpolate.interp1d(
        QMqq.X,
        QMqq.Y,
        fill_value='extrapolate',
        assume_sorted=True)
    nonParamQMCorr = interQMCorr(simul)
    # window for histograms
    minhist = -5
    maxhist = 5
    histBins = 50
    cdfBins = 100
    binning = np.linspace(minhist, maxhist, histBins)
    cdfbinning = np.linspace(minhist, maxhist, cdfBins)

    # pdf histograms
    fig, ax = plt.subplots()
    # data
    ax.hist(data,
            bins=binning,
            color=dataColour,
            density=True,
            histtype='step',
            label='data')

    # simulation
    ax.hist(simul,
            bins=binning,
            density=True,
            histtype='step',
            color=simulColour,
            label='simulation')

    # exact QM
    ax.hist(exactQMCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=exactColour,
            label='Exact QM')
    # QM qq interpolated
    ax.hist(nonParamQMCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=approxColour,
            label='Interpolated  QM qq ')
    ax.legend(loc='best')
    ax.set(xlabel='x', ylabel='pdf(x)')
    ax.set_title("Compare pdf ")
    fig.savefig("comparePdf.png", dpi=300)

    # cdf histograms
    fig, ax = plt.subplots()
    # data
    ax.hist(data,
            bins=cdfbinning,
            cumulative=1,
            color=dataColour,
            density=True,
            histtype='step',
            label='data')
    # simulation
    ax.hist(simul,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=simulColour,
            label='simulation')

    # exact QM
    ax.hist(exactQMCorr,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=exactColour,
            label='Exact QM')
    # QM qq interpolated
    ax.hist(nonParamQMCorr,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=approxColour,
            label='Interpolated  QM qq ')
    ax.legend(loc='upper left')
    ax.set(xlabel='x', ylabel='cdf(x)')
    ax.set_title("Compare CDF ")
    fig.savefig("compareCDF.png", dpi=300)


if __name__ == "__main__":
    testExampleInterpolation()
