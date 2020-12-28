import numpy as np
import matplotlib.pyplot as plt


def compareHist(data,
                simul,
                corrected,
                trueModel,
                binning,
                title,
                name):
    ''' Helper for pdf plots that are common among methods  '''

    dataColour = 'black'
    simulColour = 'skyblue'
    correctionColour = 'crimson'

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

    # corrected simulation
    ax.hist(corrected,
            bins=binning,
            density=True,
            histtype='step',
            color=correctionColour,
            label='corrected')

    if trueModel is not None:
        x = np.linspace(binning[0], binning[-1], 500)
        ax.plot(x,
                trueModel.pdf(x),
                color='grey',
                linewidth=1,
                label='True Model')

    ax.legend(loc='center right')
    ax.set(xlabel='x', ylabel='pdf(x)')
    ax.set_title(title)

    fig.savefig(name, dpi=300)


def compareCDF(data,
               simul,
               corrected,
               trueModel,
               cdfbinning,
               title,
               name):
    ''' Helper for pdf plots that are common among methods  '''

    dataColour = 'black'
    simulColour = 'skyblue'
    correctionColour = 'crimson'

    fig, ax = plt.subplots()
    ax.hist(data,
            bins=cdfbinning,
            cumulative=1,
            histtype='step',
            color=dataColour,
            density=True,
            label='data')

    ax.hist(simul,
            bins=cdfbinning,
            cumulative=1,
            histtype='step',
            color=simulColour,
            density=True,
            label='simulation')

    ax.hist(corrected,
            bins=cdfbinning,
            cumulative=1,
            histtype='step',
            color=correctionColour,
            density=True,
            label='corrected')
    if trueModel is not None:
        x = np.linspace(cdfbinning[0], cdfbinning[-1], 500)
        ax.plot(x,
                trueModel.cdf(x),
                color='grey',
                linewidth=1,
                label='True Model')

    ax.legend(loc='upper left')
    ax.set(xlabel='Input value', ylabel='%')
    ax.set_title(title)

    fig.savefig(name, dpi=300,)
