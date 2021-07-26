import ROOT
import numpy as np


def convertToArray(filename):
    '''convert branch from Kamal's christmas file
    to numpy'''
    rootFile = ROOT.TFile.Open(filename)
    tree = rootFile.Get("subtree")
    list_tmp = []
    for event in tree:
        v = event.sigma_qp
        for i in v:
            list_tmp.append(i)
    tmp = np.array(list_tmp)
    sort_tmp = np.sort(tmp)
    low = np.percentile(sort_tmp, 0.05)
    up = np.percentile(sort_tmp, 99.95)
    return sort_tmp[(sort_tmp > low) & (sort_tmp < up)]


def correction(inputMC, inputData):

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

    # check that the 2 first moments are now close
    print("original MC mean ", np.mean(inputMC),
          " inputData mean ", np.mean(inputData),
          " corrected mean", np.mean(corrected))
    print("original MC sigma ", np.std(inputMC),
          " inputData sigma ", np.std(inputData),
          " corrected sigma", np.std(corrected))

    # plotting things ...
    # pick some reasonable min and max using
    # the input entries
    hMax = max(max(inputMC), max(inputData))
    hMin = min(min(inputMC), min(inputData))

    hMC = ROOT.TH1F("hMC", "hMC", 100, hMin, hMax)
    hinputData = ROOT.TH1F("hinputData", "hinputData", 100, hMin, hMax)
    hLSCorrected = ROOT.TH1F("hLSCorrected", "hLSCorrected", 100, hMin, hMax)
    hVARICorrected = ROOT.TH1F(
        "hVARICorrected", "hVARICorrected", 100, hMin, hMax)

    ROOT.gStyle.SetOptStat(ROOT.kFALSE)
    ROOT.gStyle.SetOptTitle(ROOT.kFALSE)
    c1 = ROOT.TCanvas("c1", "c1")
    c1.cd()

    for i in np.nditer(inputMC):
        hMC.Fill(i)

    for i in np.nditer(inputData):
        hinputData.Fill(i)

    for i in np.nditer(LSCorrected):
        hLSCorrected.Fill(i)

    for i in np.nditer(corrected):
        hVARICorrected.Fill(i)
    # colors ....
    hinputData.SetMarkerColor(ROOT.kBlack)
    hinputData.SetMarkerStyle(ROOT.kFullCircle)
    hMC.SetLineColor(ROOT.kRed+1)
    hMC.SetLineWidth(3)
    hVARICorrected.SetLineColor(ROOT.kSpring-6)
    hVARICorrected.SetLineWidth(3)
    # Draw
    hMC.Scale(hinputData.Integral()/hMC.Integral())
    hMC.Draw("histSAME")
    hVARICorrected.Scale(hinputData.Integral()/hVARICorrected.Integral())
    hVARICorrected.Draw("histSAME")
    hinputData.Draw("ESAME")
    # Legend
    legend = ROOT.TLegend(0.5, 0.7, 0.9, 0.9)
    legend.SetHeader("correction test", "C")
    legend.AddEntry(hMC, "MC")
    legend.AddEntry(hinputData, "data")
    legend.AddEntry(hVARICorrected, "Shift and Variance Corrected")
    legend.Draw()

    c1.SaveAs("applyVARI.pdf")


if __name__ == "__main__":
    MC = convertToArray("MC_eta0.4_0.8_pt30_45_psec5_CB.root")
    DATA = convertToArray("DATA_eta0.4_0.8_pt30_45_psec5_CB.root")
    print("MC entries", len(MC))
    print("DATA entries", len(DATA))
    correction(MC, DATA)
