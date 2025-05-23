import numpy as np
import scipy.stats
import scipy.interpolate
import matplotlib.pyplot as plt
from QuantileMapping.QMqqMap import (
    QMqqMap)
from QuantileMapping.ParametricQM import (
    parametricQM)
from VarianceScaling.Scale import getCorrected

def testExampleInterpolation():
    '''
    Create a nominal and a distorted distribution
    Study :
     - exact parametric QM since we know the exact true model
     - non parametric QM pretending we do not know the true model
    '''
    shift = 0.7
    smear = 2
    NumData = 10000
    NumSimul = 10000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)
    # Do pefect parametric correction
    exactQMCorr = parametricQM(simul, trueModel, distortedModel)
    # Do shift and Variance correction
    variCorr = getCorrected(simul, data)
    # Do non-parametric QM correction
    numPoints = 50
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=1,
        endPerc=99,
        numPoints=numPoints,
        sigma=1.96)
    # uncertainty treatment for the qq map
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
    # Use a spline weight by the error of the points
    splineRep = scipy.interpolate.splrep(
        QMqq.X, QMqq.Y, w=1.0/np.sqrt(VY), k=2)
    knots = splineRep[0]
    coeff = splineRep[1]
    degree = splineRep[2]
    spline = scipy.interpolate.BSpline(
        t=knots,
        c=coeff,
        k=degree)
    # Do not parametric using the spline
    nonParamQMCorr = spline(simul)

    # Fix colours for plotting
    lineColour = 'black'
    dataColour = 'black'
    exactColour = 'skyblue'
    approxColour = 'red'
    simulColour = 'forestgreen'
    variColour = 'purple'

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

    # histogram definitions.
    sortedsimul = np.sort(simul)
    sorteddata = np.sort(data)
    minhist = min(sortedsimul[0], sorteddata[0])
    maxhist = max(sortedsimul[-1], sorteddata[-1])
    histBins = 40
    cdfBins =  40
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
            label='Data')
    # simulation
    ax.hist(simul,
            bins=binning,
            density=True,
            histtype='step',
            color=simulColour,
            label='Simulation')
    # exact QM
    ax.hist(exactQMCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=exactColour,
            label='Exact QM')
    # Variance and scaling
    ax.hist(variCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=variColour,
            label='Scale/Stretch')
    # QM qq interpolated
    ax.hist(nonParamQMCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=approxColour,
            label='qq map')
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
    # Variance and scaling
    ax.hist(variCorr,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=variColour,
            label='Scale/Stretch')
    # QM qq interpolated
    ax.hist(nonParamQMCorr,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=approxColour,
            label='qq map')
    ax.legend(loc='best')
    ax.set(xlabel='x', ylabel='cdf(x)')
    ax.set_title("Compare CDF ")
    fig.savefig("compareCDF.png", dpi=300)


if __name__ == "__main__":
    testExampleInterpolation()
