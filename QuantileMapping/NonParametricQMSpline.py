import numpy as np


class QMqqMap:
    '''
      Create a q-q map   
      percentiles(simul,target percentage)
      percentiles(data,target percentage)

      Each value returned by uncorrected[i]
      is mapped to the values returned by
      nominal[i]

      Inputs :
        data : data input array

        simul : simulation input array.

        targetPerc : sequence of target percentage to compute
        percentiles for during the mapping
        (array with shorted numbers in the range 0,100)

        numBootstrap : Number of bootstrap samples
        if <=1 no bootstrap (default 2000)

        percInterpolation: interpolation option when calculating percentiles

      Notes:
      The grid of uncorrected values input,
      for which  a correction is estimated, is defined by
      input = percentiles(simul,target percentages)

      If numBootstrap <=1
      correction = percentiles(data, target percentages)

      If  numBootstrap >1 then the data are resampled
      and
      correction^tilda = percentiles(data_resample, target percentages)
      is calcualated.
      The correction^tilda from all data bootstraps
      are used to derive the 95% interval for correction^hat.
    '''

    def __init__(
        self,
        data,
        simul,
        targetPerc,
        bootstrapMode='data',
        numBootstrap=2000,
        percInterpolation='linear'
    ):

        # The percentiles we will use as x values.
        self.uncorrected = np.percentile(
            simul, q=targetPerc,
            interpolation=percInterpolation)
        # Initialize  outputs to 0
        numPercentiles = targetPerc.size
        self.nominal = np.zeros(numPercentiles)
        self.up = np.zeros(numPercentiles)
        self.down = np.zeros(numPercentiles)

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
            self.nominal[i] = 0.5 * (down+up)
            self.down[i] = down
            self.up[i] = up

    def _bootstrapNone(
            self,
            data,
            simul,
            targetPerc,
            percInterpolation):
        ''' Helper method when no bootstrap is used up=nominal=down'''
        self.nominal = np.percentile(
            data,
            q=targetPerc,
            interpolation=percInterpolation)
        self.up = self.nominal
        self.down = self.nominal

    def savetxt(self, name):
        ''' save the result in a txt (csv) file
            columns are
            input , nominal , down , up
            variations.
        '''
        result = np.column_stack((
            self.uncorrected,
            self.nominal,
            self.down,
            self.up))
        np.savetxt(name, result)
