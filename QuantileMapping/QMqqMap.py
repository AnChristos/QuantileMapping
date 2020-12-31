import numpy as np
from QuantileMapping.QuantileHelper import npCI


class QMqqMap:
    '''
      Create a q-q map
      X = percentiles(simul,target percentage)
      Y =  percentiles(data,target percentage)

      Each value returned by X[i]
      is mapped to the values returned by
      Y[i]
      X = percentiles(simul,target percentages)
      Y = percentiles(data, target percentages)
      Inputs :
        y : input array to map to Y

        x : input array to map to X

        startPerc  : 1st point in the map will be:
        (startPerc(simul),startPerc(data) ).

        endPerc : last point ine the map will be:
        (endPerc(simul),endPerc(data) ).

        numPoint : Number of points in the q-q map

        Note : Extreme quantile values might be inaccurate
    '''

    def __init__(
        self,
        x,
        y,
        startPerc,
        endPerc,
        numPoints
    ):
        _percEps = 1./min(len(x), len(y))
        _startPerc = _percEps*100
        _endPerc = 100. - _startPerc
        if(startPerc < _startPerc):
            raise ValueError("start percentile target smaller"
                             " than 1.0/max(len(x),len(y))")
        else:
            _startPerc = startPerc

        if(endPerc > _endPerc):
            raise ValueError("end percentile target larger"
                             " than 1 - 1.0/min(len(x),len(y))")
        else:
            _endPerc = endPerc

        targetPerc = np.linspace(
            _startPerc, _endPerc, numPoints)
        sortx = np.sort(x)
        sorty = np.sort(y)
        # The percentiles we will use as x values.
        self.X = np.percentile(
            sortx, q=targetPerc)
        self.Y = np.percentile(
            sorty, q=targetPerc)
        self.Xup, self.Xlow = npCI(
            sortx,
            targetPerc/100.,
            assume_sorted=True)
        self.Yup, self.Ylow = npCI(
            sorty,
            targetPerc/100.,
            assume_sorted=True)

    def savetxt(self, name):
        ''' save the result in a txt (csv) file
            columns are
            input , nominal , down , up
            variations.
        '''
        result = np.column_stack((
            self.X,
            self.Y,
            self.Ydown,
            self.Yup))
        np.savetxt(name, result)
