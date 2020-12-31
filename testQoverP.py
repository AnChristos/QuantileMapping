import numpy as np
import matplotlib.pyplot as plt
import scipy.odr
from QuantileMapping.QMqqMap import (
    QMqqMap)


def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]


def testQoverP():
    '''
    Test when simul is shifted by a constant amount
    '''
    # Lets create some data and a simul that does describe
    # an effect so is shifted
    data = np.loadtxt("Data_eta0.4_0.8_pt30_45_psec5_CB.txt")
    simul = np.loadtxt("MC_eta0.4_0.8_pt30_45_psec5_CB.txt")
    numPoints = 100
    # Do non-parametric QM correction
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=1,
        endPerc=99,
        numPoints=numPoints)

    QMqq.savetxt("qqPlot_qoverP.txt")

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

    approxColour = 'red'
    lineColour = 'black'
    approxColour = 'red'
    dataColour = 'black'
    simulColour = 'forestgreen'
    fig, ax = plt.subplots()
    ax.plot(QMqq.X,
            f(output.beta, QMqq.X),
            color=approxColour,
            label='Fit of  QM q-q map')
    ax.errorbar(x=QMqq.X,
                y=QMqq.Y,
                xerr=errorX,
                yerr=errorY,
                marker='.',
                ls='none',
                markersize=2,
                color=lineColour,
                label='q-q map points')
    ax.legend(loc='best')
    string1 = (
        'slope = {:.5f} +- {:.5f}'
        .format(output.beta[0], output.sd_beta[0]))
    string2 = (
        'intersept = {:.5f} +- {:.5f}'
        .format(output.beta[1], output.sd_beta[1]))
    ax.text(x=0.018, y=0.016, s=string1)
    ax.text(x=0.018, y=0.015, s=string2)
    ax.set(xlabel='Input ', ylabel='Corrected input')
    ax.set_title('Fitted line on q-q map vs Perfect QM Correction')
    fig.savefig('qqMapFit_relSigmaQoverP.png', dpi=300)

    # corrected simul
    nonParamQMCorr = f(output.beta, simul)
    # window for histograms
    minhist = 0.012
    maxhist = 0.025
    histBins = 100
    binning = np.linspace(minhist, maxhist, histBins)

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
    fig.savefig("comparePdf_relSigmaQoverP.png", dpi=300)


if __name__ == "__main__":
    testQoverP()
