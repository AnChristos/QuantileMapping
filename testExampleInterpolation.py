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
    NumData = 40000
    NumSimul = 80000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)
    numPoints = 50
    # Do non-parametric QM correction
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=1,
        endPerc=99,
        numPoints=numPoints,
        sigma=1.96)

    # Fix colours for plotting
    lineColour = 'black'
    dataColour = 'black'
    exactColour = 'skyblue'
    approxColour = 'red'
    simulColour = 'forestgreen'

    # uncertainty treatment
    Xminus = QMqq.X - QMqq.Xlow
    Xplus = QMqq.Xup - QMqq.X
    errorX = np.row_stack((Xminus, Xplus))

    Yminus = QMqq.Y - QMqq.Ylow
    Yplus = QMqq.Yup - QMqq.Y
    errorY = np.row_stack((Yminus, Yplus))

    # Asymmetric Errors , Roger Barlow, PHYSTAT2003
    meanSigmaY = (Yplus + Yminus) * 0.5
    diffSigmaY = (Yplus - Yminus) * 0.5
    VY = meanSigmaY * meanSigmaY + 2 * diffSigmaY * diffSigmaY

    splineRep = scipy.interpolate.splrep(
        QMqq.X, QMqq.Y, w=1.0/np.sqrt(VY), k=3)
    knots = splineRep[0]
    coeff = splineRep[1]
    degree = splineRep[2]
    spline = scipy.interpolate.BSpline(
        t=knots,
        c=coeff,
        k=degree)

    fig, ax = plt.subplots()
    ax.errorbar(x=QMqq.X,
                y=QMqq.Y,
                xerr=errorX,
                yerr=errorY,
                marker='.',
                ls='none',
                markersize=2,
                color=approxColour,
                label='q-q map points')
    ax.plot(QMqq.X,
            spline(QMqq.X),
            color=lineColour,
            label='Smooth spline q-q map')
    ax.plot(QMqq.X,
            parametricQM(QMqq.X, trueModel, distortedModel),
            color=exactColour,
            label='Exact QM')
    ax.legend(loc='best')
    ax.set(xlabel='Input ', ylabel='Corrected input')
    ax.set_title('Fitted line on q-q map vs Perfect QM Correction')
    fig.savefig('qqMapSpline.png', dpi=300)

    # Do not parametric using the spline
    nonParamQMCorr = spline(simul)
    # Do pefect parametric correction
    exactQMCorr = parametricQM(simul, trueModel, distortedModel)

    # window for histograms
    sortedsimul = np.sort(simul)
    sorteddata = np.sort(data)
    minhist = min(sortedsimul[0], sorteddata[0])
    maxhist = max(sortedsimul[-1], sorteddata[-1])
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
    fig.savefig("comparePdfInterpolation.png", dpi=300)

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
    fig.savefig("compareCDFInterpolation.png", dpi=300)


if __name__ == "__main__":
    testExampleInterpolation()
