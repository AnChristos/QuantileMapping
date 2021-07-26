import ROOT
import numpy as np
import scipy.stats
import scipy.signal


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
    return np.array(list_tmp)


def applyDistortion(inputArray):
    ''' Add a normal to each entry '''
    distortpdf = scipy.stats.norm(loc=0.13, scale=0.009)
    distortion = distortpdf.rvs(size=len(inputArray))
    return distortion, inputArray + distortion


def correction(inputMC, pseudoData, distortion):

    print("MC ", np.mean(inputMC), np.std(inputMC))
    print("DATA ", np.mean(pseudoData), np.std(pseudoData))
    print("distortion ", np.mean(distortion), np.std(distortion))
    hMax = max(max(distortion), max(pseudoData), max(inputMC))
    hMin = min(min(distortion), min(pseudoData), min(inputMC))
    numBins = 200
    hMC = ROOT.TH1F("hMC", "hMC", numBins, hMin, hMax)
    hPseudoData = ROOT.TH1F("hPseudoData", "hPseudoData", numBins, hMin, hMax)
    hdistortion = ROOT.TH1F(
        "hdistortion", "hdistortion", numBins, hMin, hMax)
    c1 = ROOT.TCanvas("c1", "c1")
    c1.cd()

    for i in np.nditer(inputMC):
        hMC.Fill(i)

    for i in np.nditer(pseudoData):
        hPseudoData.Fill(i)

    for i in np.nditer(distortion):
        hdistortion.Fill(i)

    # colors ....
    hMC.SetLineColor(ROOT.kRed+1)
    hPseudoData.SetLineColor(ROOT.kBlack)
    hMC.Draw()
    hPseudoData.Draw("SAME")
    hdistortion.DrawCopy("SAME")

    hConvolved = ROOT.TH1F("hConvolved", "hConvolved", numBins, hMin, hMax)
    hdistortion.Scale(1.0/hdistortion.Integral())

    for n in range(0, numBins):
        sum = 0
        x_n = hMC.GetXaxis().GetBinCenter(n+1)
        
        for m in range(0, numBins):

            x_m = hMC.GetXaxis().GetBinCenter(m+1)
            x_n_min_m = x_n - x_m
            bin_n = hMC.GetXaxis().FindBin(x_n) # this is not really necessary, the result will be n
            bin_m = hMC.GetXaxis().FindBin(x_m) # this is not really necessary, the result will be m
            bin_n_min_m = hMC.GetXaxis().FindBin(x_n_min_m) # this correctly takes into account the offset of the histogram!

            f_tau = hMC.GetBinContent(bin_m)
            g_t_m_tau = hdistortion.GetBinContent(bin_n_min_m)
            sum += f_tau * g_t_m_tau
        hConvolved.SetBinContent(n+1, sum)

    print("Convolution", hConvolved.GetMean(), hConvolved.GetRMS())
    hConvolved.SetLineColor(ROOT.kSpring-6)
    hConvolved.Draw("SAME")
    c1.SaveAs("test.pdf")


if __name__ == "__main__":
    MC = convertToArray("/afs/cern.ch/user/c/christos/public/Quantile_mapping_play/MC_eta0.4_0.8_pt30_45_psec5_CB.root")
    print("entries", len(MC))
    distortion, pseudoData = applyDistortion(MC)
    correction(MC, pseudoData, distortion)
