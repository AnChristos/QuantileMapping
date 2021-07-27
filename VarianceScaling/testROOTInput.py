import ROOT
import numpy as np
import scipy.stats


def convertToArray(filename):
    '''convert branch from Kamal's christmas file
    to numpy'''
    rootFile = ROOT.TFile.Open(filename)
    tree = rootFile.Get("MC_eta0_0.4_pt50_60_phi-0.965_-0.605_CB")
    list_tmp = []
    for event in tree:
        v = event.sigma_qp
        for i in v:
            list_tmp.append(i)
    tmp = np.array(list_tmp)
    sort_tmp = np.sort(tmp)
    low = np.percentile(sort_tmp, 0.25)
    up = np.percentile(sort_tmp, 99.75)
    return sort_tmp[(sort_tmp > low) & (sort_tmp < up)]


def applyDistortion(inputArray):
    ''' Add a normal to each entry '''
    distortion = scipy.stats.norm(loc=0.001, scale=0.0015)
    return inputArray + distortion.rvs(size=len(inputArray))


def correction(inputMC, pseudoData):

    # linear scaling
    mean_data = np.mean(pseudoData)
    mean_simul = np.mean(inputMC)
    estShift = mean_data - mean_simul
    # up to here we correct just a shift
    LSCorrected = inputMC + estShift

    # Let's perform a simplified smearing
    # shift everything to 0 mean
    mean_shifted_simul = np.mean(LSCorrected)
    zero_mean_simul = LSCorrected - mean_shifted_simul
    # And then calculate the ratio of the data simul sigma
    sigma_data = np.std(pseudoData)
    sigma_simul = np.std(zero_mean_simul)
    sigma_ratio = sigma_data/sigma_simul
    # The final corrected one
    corrected = zero_mean_simul * sigma_ratio + mean_shifted_simul

    # check that the 2 first moments are now close
    print("original MC mean ", np.mean(inputMC),
          " pseudodata mean ", np.mean(pseudoData),
          " corrected mean", np.mean(corrected))
    print("original MC sigma ", np.std(inputMC),
          " pseudodata sigma ", np.std(pseudoData),
          " corrected sigma", np.std(corrected))

    # plotting things ...
    # pick some reasonable min and max using
    # the input entries
    hMax = max(max(inputMC), max(pseudoData))
    hMin = min(min(inputMC), min(pseudoData))

    hMC = ROOT.TH1F("hMC", "hMC", 100, hMin, hMax)
    hPseudoData = ROOT.TH1F("hPseudoData", "hPseudoData", 100, hMin, hMax)
    hLSCorrected = ROOT.TH1F("hLSCorrected", "hLSCorrected", 100, hMin, hMax)
    hVARICorrected = ROOT.TH1F(
        "hVARICorrected", "hVARICorrected", 100, hMin, hMax)
    c1 = ROOT.TCanvas("c1", "c1")
    c1.cd()

    for i in np.nditer(inputMC):
        hMC.Fill(i)

    for i in np.nditer(pseudoData):
        hPseudoData.Fill(i)

    for i in np.nditer(LSCorrected):
        hLSCorrected.Fill(i)

    for i in np.nditer(corrected):
        hVARICorrected.Fill(i)
    # colors ....
    hMC.SetLineColor(ROOT.kRed+1)
    hPseudoData.SetLineColor(ROOT.kBlack)
    hLSCorrected.SetLineColor(ROOT.kBlue+1)
    hVARICorrected.SetLineColor(ROOT.kSpring-6)
    hMC.Draw()
    hPseudoData.Draw("SAME")
    # hLSCorrected.Draw("SAME")
    hVARICorrected.Draw("SAME")

    c1.SaveAs("testVARI.pdf")


if __name__ == "__main__":
    MC = convertToArray("MC_eta0_0.4_pt50_60_phi-0.965_-0.605_CB.root")
    print("entries", len(MC))
    pseudoData = applyDistortion(MC)
    correction(MC, pseudoData)
