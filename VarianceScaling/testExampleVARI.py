import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def testExampleFit():
    '''
    Create a nominal and a distorted distribution
    '''
    NumData = 100000
    trueModel = scipy.stats.norm(loc=0, scale=0.9)
    distortion = scipy.stats.norm(loc=1.0, scale=0.9)
    simul = trueModel.rvs(size=NumData)
    distorted = distortion.rvs(size=NumData)
    data = simul + distorted

    # variance scaling
    mean_data = np.mean(data)
    mean_simul = np.mean(simul)
    estShift = mean_data - mean_simul

    # up to here we correct just a shift
    meanCorrectedSimul = simul + estShift

    # Let's perform a simplified smearing
    # shift everything to 0 mean
    mean_shifted_simul = np.mean(meanCorrectedSimul)
    zero_mean_simul = meanCorrectedSimul - mean_shifted_simul
    # And then calculate the ratio of the data simul sigma
    sigma_data = np.std(data)
    sigma_simul = np.std(zero_mean_simul)
    sigma_ratio = sigma_data/sigma_simul
    # The final corrected one
    corrected = zero_mean_simul * sigma_ratio + mean_shifted_simul

    # check that the 2 first moments are now close
    print("original MC mean ", np.mean(simul),
          " pseudodata mean ", np.mean(data),
          " corrected mean", np.mean(corrected))
    print("original MC sigma ", np.std(simul),
          " pseudodata sigma ", np.std(data),
          " corrected sigma", np.std(corrected))
    # Plotting follows ....
    # Fix colours for plotting
    dataColour = 'black'
    VARIColour = 'red'
    simulColour = 'forestgreen'

    histBins = 100
    # window for histograms
    sortedsimul = np.sort(simul)
    sorteddata = np.sort(data)
    minhist = min(sortedsimul[0], sorteddata[0])
    maxhist = max(sortedsimul[-1], sorteddata[-1])
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

    # corrected simulation
    ax.hist(corrected,
            bins=binning,
            density=True,
            histtype='step',
            color=VARIColour,
            label='mean and variance corrrected simulation ')
    ax.legend(loc='best')
    ax.set(xlabel='x', ylabel='pdf(x)')
    ax.set_ylim([0, 0.6])
    ax.set_title("Compare pdf ")
    fig.savefig("comparePdf.png", dpi=300)


if __name__ == "__main__":
    testExampleFit()
