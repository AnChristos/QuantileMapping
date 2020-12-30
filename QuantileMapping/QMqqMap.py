import numpy as np


class QMqqMap:
    '''
      Create a q-q map
      X = percentiles(simul,target percentage)
      Y= percentiles(data,target percentage)

      Each value returned by X[i]
      is mapped to the values returned by
      Y[i]

      Inputs :
        data : data input array

        simul : simulation input array.

        numPoint = 500: Number of points in the q-q map

        startPerc = 0.0 : 1st point will be
        (startPerc(simul),startPerc(data) )

        endPerc = 100 : last point will be
        (endPerc(simul),endPerc(data) )

        numBootstrap = 2000 : Number of bootstrap samples
        if <=1 no bootstrap.

        percInterpolation: interpolation option when calculating percentiles

      Notes:
      X = percentiles(simul,target percentages)

      If numBootstrap <=1
      Y = percentiles(data, target percentages)

      If  numBootstrap >1 then the data are resampled
      and
      Y^tilda = percentiles(data_resample, target percentages)
      is calcualated.
      The Y^tilda from all data bootstraps
      are used to derive the 95% interval for Y.
    '''

    def __init__(
        self,
        data,
        simul,
        numPoints=500,
        startPerc=0.0,
        endPerc=100,
        bootstrapMode='data',
        numBootstrap=2000,
        percInterpolation='linear'
    ):
        targetPerc = np.linspace(
            startPerc, endPerc, numPoints)
        numPercentiles = len(targetPerc)
        # The percentiles we will use as x values.
        self.X = np.percentile(
            simul, q=targetPerc,
            interpolation=percInterpolation)
        self.Y = np.zeros(numPercentiles)
        self.Yup = np.zeros(numPercentiles)
        self.Ydown = np.zeros(numPercentiles)

        if numBootstrap > 1:
            self._bootstrapData(
                data,
                simul,
                numBootstrap,
                targetPerc,
                numPercentiles,
                percInterpolation)
        else:
            self._bootstrapNone(
                data,
                simul,
                targetPerc,
                percInterpolation)

    def _bootstrapData(
            self,
            data,
            simul,
            numBootstrap,
            targetPerc,
            numPercentiles,
            percInterpolation):
        '''
        Helper method for running with bootstrap on data
        '''
        # Array to keep track of the bootstrapss
        bootstrapResults = np.zeros(shape=(numBootstrap, numPercentiles))
        lenData = len(data)
        # create bootstraps
        for i in range(numBootstrap):
            # resample the inputs with replacement with random noise
            iData = np.random.choice(data, lenData, replace=True)
            bootstrapResults[i] = np.percentile(
                iData,
                q=targetPerc,
                interpolation=percInterpolation)

        # down,nominal,up
        quant = np.array([2.5, 97.5])
        for i in range(numPercentiles):
            down,  up = np.percentile(
                bootstrapResults[:, i],
                q=quant,
                interpolation=percInterpolation)
            self.Y[i] = 0.5 * (down+up)
            self.Ydown[i] = down
            self.Yup[i] = up

    def _bootstrapNone(
            self,
            data,
            simul,
            targetPerc,
            percInterpolation):
        ''' Helper method when no bootstrap is used up=nominal=down'''
        self.Y = np.percentile(
            data,
            q=targetPerc,
            interpolation=percInterpolation)
        self.Yup = self.Y
        self.Ydown = self.Y

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
