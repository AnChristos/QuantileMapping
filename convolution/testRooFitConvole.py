import ROOT
import numpy as np
import scipy.stats
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
    low = np.percentile(sort_tmp, 0.5)
    up = np.percentile(sort_tmp, 99.5)
    return sort_tmp[(sort_tmp > low) & (sort_tmp < up)]


def applyDistortion(inputArray):
    ''' Add a normal to each entry '''
    distortpdf = scipy.stats.norm(loc=0.001, scale=0.0015)
    distortion = distortpdf.rvs(size=len(inputArray))
    return distortion, inputArray + distortion


def correction(inputMC, pseudoData, distortion):

    hMax = max(max(pseudoData), max(inputMC))
    hMin = min(min(pseudoData), min(inputMC))
    numBins = 100
    hMC = ROOT.TH1F("hMC", "hMC", numBins, hMin, hMax)
    hPseudoData = ROOT.TH1F("hPseudoData", "hPseudoData", numBins, hMin, hMax)
    # hPseudoData.Sumw2()
    for i in np.nditer(inputMC):
        hMC.Fill(i)
    for i in np.nditer(pseudoData):
        hPseudoData.Fill(i)

    # set up a full convolution model
    # Let's get some initial Guesses for the mean
    # and the sigma of the Gauss
    estShift = hPseudoData.GetMean() - hMC.GetMean()
    sigmaMC = hMC.GetRMS()
    sigmaData = hPseudoData.GetRMS()
    estSmear = math.sqrt(sigmaData*sigmaData-sigmaMC*sigmaMC)
    # variable for the observable
    x = ROOT.RooRealVar("x", "x", hMin, hMax)
    # data (what we want to fit to)
    data = ROOT.RooDataHist("data", "data", x, hPseudoData)
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
    c1 = ROOT.TCanvas("c1", "c1")
    c1.cd()
    xframe = x.frame()
    data.plotOn(xframe)
    hRooMC.plotOn(xframe,
                  ROOT.RooFit.DrawOption("C"),
                  ROOT.RooFit.LineColor(ROOT.kRed))
    histXGaus.plotOn(xframe,
                     ROOT.RooFit.LineColor(ROOT.kSpring-6))
    xframe.Draw()
    c1.SaveAs("testFit.pdf")


if __name__ == "__main__":
    MC = convertToArray("MC_eta0.4_0.8_pt30_45_psec5_CB.root")
    print("entries", len(MC))
    distortion, pseudoData = applyDistortion(MC)
    correction(MC, pseudoData, distortion)
