import ROOT
import numpy as np
import math


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

    hMax = max(max(inputData), max(inputMC))
    hMin = min(min(inputData), min(inputMC))
    numBins = 100
    hMC = ROOT.TH1F("hMC", "hMC", numBins, hMin, hMax)
    hinputData = ROOT.TH1F("hinputData", "hinputData", numBins, hMin, hMax)
    # hinputData.Sumw2()
    for i in np.nditer(inputMC):
        hMC.Fill(i)
    for i in np.nditer(inputData):
        hinputData.Fill(i)

    # set up a full convolution model
    # Let's get some initial Guesses for the mean
    # and the sigma of the Gauss
    estShift = hinputData.GetMean() - hMC.GetMean()
    sigmaMC = hMC.GetRMS()
    sigmaData = hinputData.GetRMS()
    estSmear = math.sqrt(sigmaData*sigmaData-sigmaMC*sigmaMC)
    # variable for the observable
    x = ROOT.RooRealVar("x", "qOverP", hMin, hMax)
    # data (what we want to fit to)
    data = ROOT.RooDataHist("data", "data", x, hinputData)
    # create an probability density function
    # for the MC
    hRooMC = ROOT.RooDataHist("hRooMC", "hRooMC", x, hMC)
    hpdfMC = ROOT.RooHistPdf("hpdfMC", "hpdfMC", x, hRooMC, 0)

    # A Gaussian we convolve with
    meanG = ROOT.RooRealVar("meanG", "meanG", estShift,
                            0.5*estShift, 1.5*estShift)
    sigmaG = ROOT.RooRealVar(
        "sigmaG", "sigmaG",
        estSmear, 0.0, 1.5 * estSmear)
    gauss = ROOT.RooGaussian("gauss", "gauss", x, meanG, sigmaG)

    # The actual convolution model
    histXGaus = ROOT.RooFFTConvPdf(
        "histXGaus", "histo (X) gauss", x, hpdfMC, gauss)
    # Let's fit the model to the data so as to get
    # the parameters for the Gauss
    histXGaus.fitTo(data)
    print("fitted mean", meanG.getVal())
    print("fitted sigma ", sigmaG.getVal())

    # Some plotting ...
    ROOT.gStyle.SetOptStat(ROOT.kFALSE)
    ROOT.gStyle.SetOptTitle(ROOT.kFALSE)
    c1 = ROOT.TCanvas("c1", "c1")
    c1.cd()
    xframe = x.frame()
    data.plotOn(xframe,
                ROOT.RooFit.Name("dataPoints"))
    hpdfMC.plotOn(xframe,
                  ROOT.RooFit.LineColor(ROOT.kRed),
                  ROOT.RooFit.Name("MCpdf"))
    histXGaus.plotOn(xframe,
                     ROOT.RooFit.LineColor(ROOT.kSpring-6),
                     ROOT.RooFit.Name("Convolutionpdf"))
    xframe.Draw()
    legend = ROOT.TLegend(0.5, 0.7, 0.9, 0.9)
    legend.SetHeader("correction test", "C")
    legend.AddEntry(xframe.findObject("MCpdf"), "MC")
    legend.AddEntry(xframe.findObject("dataPoints"), "data")
    legend.AddEntry(xframe.findObject("Convolutionpdf"),
                    "Convolution Corrected")
    legend.Draw()
    c1.SaveAs("applyFit.pdf")


if __name__ == "__main__":
    MC = convertToArray("MC_eta0.4_0.8_pt30_45_psec5_CB.root")
    DATA = convertToArray("DATA_eta0.4_0.8_pt30_45_psec5_CB.root")
    print("MC entries", len(MC))
    print("DATA entries", len(DATA))
    correction(MC, DATA)
