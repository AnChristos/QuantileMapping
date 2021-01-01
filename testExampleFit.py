import numpy as np
import scipy.stats
import scipy.interpolate
import scipy.odr
import matplotlib.pyplot as plt
from QuantileMapping.QMqqMap import (
    QMqqMap)
from QuantileMapping.ParametricQM import (
    parametricQM)


def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


def testExampleFit():
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
    meanSigmaX = (Xplus + Xminus) * 0.5
    diffSigmaX = (Xplus - Xminus) * 0.5
    VX = meanSigmaX * meanSigmaX + 2 * diffSigmaX * diffSigmaX
    meanSigmaY = (Yplus + Yminus) * 0.5
    diffSigmaY = (Yplus - Yminus) * 0.5
    VY = meanSigmaY * meanSigmaY + 2 * diffSigmaY * diffSigmaY

    # fit a straight line
    x1forguess = QMqq.X[int(numPoints*0.3)]
    x2forguess = QMqq.X[int(numPoints*0.7)]
    y1forguess = QMqq.Y[int(numPoints*0.3)]
    y2forguess = QMqq.Y[int(numPoints*0.7)]
    slopeguess = (y2forguess-y1forguess)/(x2forguess-x1forguess)
    constguess = y2forguess - slopeguess * x2forguess
    guess = np.array([slopeguess, constguess])

    linear = scipy.odr.Model(f)
    fitdata = scipy.odr.Data(QMqq.X, QMqq.Y, wd=1./VX, we=1./VY)
    odr = scipy.odr.ODR(fitdata, linear, beta0=guess)
    output = odr.run()
    output.pprint()

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
            f(output.beta, QMqq.X),
            color=lineColour,
            label='Fit of  QM q-q map')
    ax.plot(QMqq.X,
            parametricQM(QMqq.X, trueModel, distortedModel),
            color=exactColour,
            label='Exact QM')
    ax.legend(loc='best')
    string1 = (
        'slope = {:.4f} +- {:.4f}'
        .format(output.beta[0], output.sd_beta[0]))
    string2 = (
        'intersept = {:.4f} +- {:.4f}'
        .format(output.beta[1], output.sd_beta[1]))
    ax.text(x=0, y=-1, s=string1)
    ax.text(x=0, y=-1.2, s=string2)
    ax.set(xlabel='Input ', ylabel='Corrected input')
    ax.set_title('Fitted line on q-q map vs Perfect QM Correction')
    fig.savefig('qqMapFit.png', dpi=300)

    # Do not parametric using the fit
    nonParamQMCorr = f(output.beta, simul)
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
    # QM qq fitted
    ax.hist(nonParamQMCorr,
            bins=binning,
            density=True,
            histtype='step',
            color=approxColour,
            label='Fitted  QM qq ')
    ax.legend(loc='best')
    ax.set(xlabel='x', ylabel='pdf(x)')
    ax.set_title("Compare pdf ")
    fig.savefig("comparePdfFit.png", dpi=300)

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

    # QM qq fitted
    ax.hist(nonParamQMCorr,
            bins=cdfbinning,
            cumulative=1,
            density=True,
            histtype='step',
            color=approxColour,
            label='Fitted  QM qq ')
    ax.legend(loc='upper left')
    ax.set(xlabel='x', ylabel='cdf(x)')
    ax.set_title("Compare CDF ")
    fig.savefig("compareCDFFit.png", dpi=300)


if __name__ == "__main__":
    testExampleFit()
