import numpy as np
import matplotlib.pyplot as plt


def compareHist(data,
                simul,
                corrected,
                minhist,
                maxhist,
                histBins,
                title,
                name):
    ''' Helper for some plots that are common among methods  '''

    dataColour = 'black'
    simulColour = 'skyblue'
    correctionColour = 'crimson'

    # Histo plots
    binning = np.linspace(minhist, maxhist, histBins)
    # data vs simulation
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
    ax.legend(loc='center right')
    ax.set(xlabel='x', ylabel='pdf(x)')
    ax.set_title(title)

    fig.savefig(name, dpi=300)
