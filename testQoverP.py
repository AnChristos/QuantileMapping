import numpy as np
import matplotlib.pyplot as plt
from QuantileMapping.QMqqMap import (
    QMqqMap)


def testQoverP():
    '''
    Test when simul is shifted by a constant amount
    '''
    # Lets create some data and a simul that does describe
    # an effect so is shifted
    data = np.loadtxt("Data_eta0.4_0.8_pt30_45_psec5_CB.txt")
    simul = np.loadtxt("MC_eta0.4_0.8_pt30_45_psec5_CB.txt")

    # Do non-parametric QM correction
    QMqq = QMqqMap(
        simul,
        data,
        startPerc=0.1,
        endPerc=99.9,
        numPoints=500)
    QMqq.savetxt("qqPlot_qoverP.txt")
    lowerErrorX = QMqq.X - QMqq.Xlow
    upperErrorX = QMqq.Xup - QMqq.X
    ErrorX = np.row_stack((lowerErrorX, upperErrorX))

    lowerErrorY = QMqq.Y - QMqq.Ylow
    upperErrorY = QMqq.Yup - QMqq.Y
    ErrorY = np.row_stack((lowerErrorY, upperErrorY))

    fig, ax = plt.subplots()
    ax.errorbar(x=QMqq.X,
                y=QMqq.Y,
                xerr=ErrorX,
                yerr=ErrorY,
                marker='.',
                ls='none',
                markersize=2,
                color='crimson',
                label='Estimated QM')
    ax.legend(loc='best')
    ax.set(xlabel='input ', ylabel='corrected input')
    ax.set_title("RelativeqQ/P correcion")

    fig.savefig("RelativeQoverP.png", dpi=300)


if __name__ == "__main__":
    testQoverP()
