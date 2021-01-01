import ROOT
import numpy as np


def converter(rootfile):
    '''To convert from Kamal's file to txt,csv  numpy
        style.'''
    rootFile = ROOT.TFile.Open(rootfile)
    tree = rootFile.Get("subtree")
    list_tmp = []
    for event in tree:
        v = event.sigma_qp
        for i in v:
            list_tmp.append(i)

    arrayMC = np.array(list_tmp)
    outName = rootfile.replace("root", "txt")
    np.savetxt(outName, arrayMC)
    rootFile.Close()


if __name__ == "__main__":
    inputs = ["Data_eta0.4_0.8_pt30_45_psec5_CB.root",
              "MC_eta0.4_0.8_pt30_45_psec5_CB.root"]
    for i in inputs:
        converter(i)
