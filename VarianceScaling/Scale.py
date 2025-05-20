import numpy as np

def getCorrected(inputMC, inputData):

    # linear scaling
    mean_data = np.mean(inputData)
    mean_simul = np.mean(inputMC)
    estShift = mean_data - mean_simul
    # up to here we correct just a shift
    LSCorrected = inputMC + estShift

    # Let's perform a simplified smearing
    # shift everything to 0 mean
    mean_shifted_simul = np.mean(LSCorrected)
    zero_mean_simul = LSCorrected - mean_shifted_simul
    # And then calculate the ratio of the data simul sigma
    sigma_data = np.std(inputData)
    sigma_simul = np.std(zero_mean_simul)
    sigma_ratio = sigma_data/sigma_simul
    # The final corrected one
    corrected = zero_mean_simul * sigma_ratio + mean_shifted_simul

    return corrected
