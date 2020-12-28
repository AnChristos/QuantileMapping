import numpy as np
import matplotlib.pyplot as plt


def compareHist(data,
                simul,
                corrected,
                trueModel,
                binning,
                title,
                name):
    ''' Helper for some plots that are common among methods  '''

    dataColour = 'black'
    simulColour = 'skyblue'
    correctionColour = 'crimson'

    # Histo plots
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
