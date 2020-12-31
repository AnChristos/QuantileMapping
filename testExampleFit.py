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
    NumData = 10000
    NumSimul = 40000
    trueModel = scipy.stats.norm()
    distortedModel = scipy.stats.norm(loc=shift, scale=smear)
    data = trueModel.rvs(size=NumData)
    simul = distortedModel.rvs(size=NumSimul)
    numPoints = 50
    # Do non-parametric QM correction
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=1.,
        endPerc=99.,
        numPoints=numPoints,
        sigma=1.96)

    Xminus = QMqq.X - QMqq.Xlow
    Xplus = QMqq.Xup - QMqq.X
    errorX = np.row_stack((Xminus, Xplus))

    Yminus = QMqq.Y - QMqq.Ylow
    Yplus = QMqq.Yup - QMqq.Y
    errorY = np.row_stack((Yminus, Yplus))

    # Fix colours for plotting
    exactColour = 'skyblue'
    approxColour = 'red'
    lineColour = 'black'

    # Compare estimated with exact QM
    fig, ax = plt.subplots()

    ax.plot(QMqq.X,
            parametricQM(QMqq.X, trueModel, distortedModel),
            color=exactColour,
            label='Exact QM')
    ax.legend(loc='best')
    ax.set(xlabel='Input ', ylabel='Corrected input')
    ax.set_title('Estimated vs Perfect QM Correction')
    fig.savefig('CorrectionCompare.png', dpi=300)

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
    data = scipy.odr.Data(QMqq.X, QMqq.Y, wd=1./VX, we=1./VY)
    odr = scipy.odr.ODR(data, linear, beta0=guess)
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


if __name__ == "__main__":
    testExampleFit()
