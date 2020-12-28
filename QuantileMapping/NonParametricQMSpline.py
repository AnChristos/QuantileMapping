import numpy as np
import scipy.interpolate
import scipy.stats
import math


class fitNonParametricQMSpline:
    '''
      Non-parametric estimation of
      x_corr = g(x_input)
      fitting a smoothing spline
      to the quantile-quantile plot
      percentiles(simul,target percentage)
      percentiles(data,target percentage)

      Each value returned by uncorrected()[i]
      is mapped to the values returned by
      corrected()[i]
      correctedUp()[i]
      correctedDown()[i]

      Univariate splines passing through pairs of
      [uncorrected()[i], corrected()[i]]
      [uncorrected()[i], correctedUp()[i]]
      [uncorrected()[i], correctedDown()[i]]
      are used to  approximate the correction
      function x_corr=g(x_input).

      Inputs :
        - data : data input array

        - simul : simulation input array.

        - targetPerc : sequence of target percentage to compute
        percentiles for during the mapping
        (array with shorted numbers in the range 0,100)

        - bootstrapMode = 'data' , 'both' , 'none' .  Use bootstrap
        for estimating statistical uncertainties.
        'both' : Both data and simulation
        'data' : Data only assumes a high statistics simul to be
         adjusted using limited data.
        'none' : No bootstrap, no uncertainties
         default ('both')

        - numBootstrap : Number of bootstrap samples (default 1000)

        - percInterpolation: interpolation option when calculating percentiles

        - smoothDegree : smoothDegree passed to
        scipy.interpolate.UnivariateSpline

        - Ext : Extrapolation option passed to
        scipy.interpolate.UnivariateSpline

      Notes:
      The grid of uncorrected values input,
      for which  a correction is estimated, is defined by
      input = percentiles(simul,target percentages)

      If bootstrapMode ='none' then
      correction = percentiles(data, target percentages)
      g =  UnivariateSpline( input, correction)
      and there are no uncertainties
      (the relevant methods return the same as the nominal)

      If bootstrapMode ='both'
      the method uses percentile smooth bootstrap
      to estimate the corrected values.
      Both data and simulation are resampled and a
      a correction^tilda for each value of the input
      is calculated via:
      g_resample = UnivariateSpline(simul_resample,data_resample )
      correction^tilda = g_resample(input).
      The correction^tilda from all data/MC bootstraps
      are used to derive the interval of correction^hat.
      The return values are:
      g_up = UnivariateSpline(input,correction^hat up)
      g_down = UnivariateSpline(input,correction^hat down)
      g_nominal = (g_up + g_down) * 0.5

      If bootstrapMode ='data' then the data are resampled
      and
      correction^tilda = percentiles(data_resample, target percentages)
      is calcualated.
      The correction^tilda from all data bootstraps
      are used to derive the interval of correction^hat.
      The return values are:
      g_up = UnivariateSpline(input,correction^hat up)
      g_down = UnivariateSpline(input,correction^hat down)
      g_nominal = (g_up + g_down) * 0.5
    '''

    def __init__(self,
                 data,
                 simul,
                 targetPerc,
                 bootstrapMode='both',
                 numBootstrap=1000,
                 percInterpolation='linear',
                 smoothDegree=3,
                 Ext=0
                 ):

        # The percentiles we will use as x values.
        self._uncorrected = np.percentile(
            simul, q=targetPerc,
            interpolation=percInterpolation)
        # Initialize  outputs to 0
        numPercentiles = targetPerc.size
        self._up = np.zeros(numPercentiles)
        self._down = np.zeros(numPercentiles)
        # Check if we do bootstrap (default is true)
        self._hasBootstrap = (True
                              if numBootstrap > 0
                              else False)

        if bootstrapMode == 'both':
            self._bootstrapBoth(data,
                                simul,
                                numBootstrap,
                                targetPerc,
                                numPercentiles,
                                percInterpolation,
                                smoothDegree,
                                Ext)
        elif bootstrapMode == 'data':
            self._bootstrapData(data,
                                simul,
                                numBootstrap,
                                targetPerc,
                                numPercentiles,
                                percInterpolation,
                                smoothDegree,
                                Ext)
        elif bootstrapMode == 'none':
            self._bootstrapNone(data,
                                simul,
                                targetPerc,
                                percInterpolation,
                                smoothDegree,
                                Ext)

        else:
            raise Exception("unexpected bootstrapMode ", bootstrapMode)

    def _bootstrapBoth(self,
                       data,
                       simul,
                       numBootstrap,
                       targetPerc,
                       numPercentiles,
                       percInterpolation,
                       smoothDegree,
                       Ext):
        '''
        Helper method for running with bootstrap on both
        data and simulation
        '''
        # Array to keep track of the bootstraps
        # As many rows as numBootstrap
        # as many colums as x inputs we want the y outputs / percentiles
        bootstrapResults = np.zeros(shape=(numBootstrap, numPercentiles))
        # calculate random noise to add
        effSigmaData = min(np.std(data), scipy.stats.iqr(
            data)/1.34)
        dataNoise = 0.9 * effSigmaData * math.pow(data.size, -0.2)
        effSigmaSim = min(np.std(simul), scipy.stats.iqr(
            simul)/1.34)
        simNoise = 0.9 * effSigmaSim * math.pow(simul.size, -0.2)
        # create bootstraps and repeat the mapping procedure
        for i in range(numBootstrap):
            # resample the inputs with replacement with random noise
            randomNoiseData = np.random.normal(
                0, dataNoise, size=data.size)
            randomNoiseSimul = np.random.normal(
                0, simNoise, size=simul.size)
            dataRepl = np.random.choice(
                data, len(data), replace=True)+randomNoiseData
            simulRepl = np.random.choice(
                simul, len(simul), replace=True)+randomNoiseSimul
            # create a mapping from the resamples
            percentileSim = np.percentile(
                simulRepl, q=targetPerc, interpolation=percInterpolation)
            percentileData = np.percentile(
                dataRepl, q=targetPerc, interpolation=percInterpolation)
            correctionRepl = scipy.interpolate.UnivariateSpline(
                percentileSim, percentileData, k=smoothDegree, ext=Ext)
            # call the resampled mapping for the same
            # simulation percentiles
            # used in the nominal to get new y values for them
            bootstrapResults[i] = correctionRepl(self._uncorrected)

        # Calculate correction_down, correction_up
        # The nominal value is in the middle of this interval
        quant = np.array([2.5, 97.5])
        for i in range(numPercentiles):
            down, up = np.percentile(bootstrapResults[:, i], q=quant)
            self._down[i] = down
            self._up[i] = up

        # Create interpolated g(x_input)
        self._upInterp = scipy.interpolate.UnivariateSpline(
            self._uncorrected, self._up, k=smoothDegree, ext=Ext)
        self._downInterp = scipy.interpolate.UnivariateSpline(
            self._uncorrected, self._down, k=smoothDegree, ext=Ext)

    def _bootstrapData(self,
                       data,
                       simul,
                       numBootstrap,
                       targetPerc,
                       numPercentiles,
                       percInterpolation,
                       smoothDegree,
                       Ext):
        '''
        Helper method for running with bootstrap on data
        '''
        # Array to keep track of the bootstraps
        # As many rows as numBootstrap
        # as many colums as x inputs we want the y outputs / percentiles
        bootstrapResults = np.zeros(shape=(numBootstrap, numPercentiles))
        # calculate random noise to add
        effSigmaData = min(np.std(data), scipy.stats.iqr(
            data)/1.34)
        dataNoise = 0.9 * effSigmaData * math.pow(data.size, -0.2)
        # create bootstraps and repeat the mapping procedure
        for i in range(numBootstrap):
            # resample the inputs with replacement with random noise
            randomNoiseData = np.random.normal(
                0, dataNoise, size=data.size)
            dataRepl = np.random.choice(
                data, len(data), replace=True)+randomNoiseData
            bootstrapResults[i] = np.percentile(
                dataRepl, q=targetPerc, interpolation=percInterpolation)

        # Calculate correction_down, correction_up
        # The nominal value is in the middle of this interval
        quant = np.array([2.5, 97.5])
        for i in range(numPercentiles):
            down, up = np.percentile(bootstrapResults[:, i], q=quant)
            self._down[i] = down
            self._up[i] = up

        # Create interpolated g(x_input)
        self._upInterp = scipy.interpolate.UnivariateSpline(
            self._uncorrected, self._up, k=smoothDegree, ext=Ext)
        self._downInterp = scipy.interpolate.UnivariateSpline(
            self._uncorrected, self._down, k=smoothDegree, ext=Ext)

    def _bootstrapNone(self, data,
                       simul, targetPerc,
                       percInterpolation,
                       smoothDegree, Ext):
        ''' Helper method when no bootstrap is used up=nominal=down'''
        self._up = np.percentile(
            data, q=targetPerc,
            interpolation=percInterpolation)
        self._down = self._up
        # Create interpolated g(x_input)
        self._upInterp = scipy.interpolate.UnivariateSpline(
            self._uncorrected, self._up, k=smoothDegree, ext=Ext)
        self._downInterp = self._upInterp

    def uncorrected(self):
        ''' Return x_input uncorrected values a x_corr was estimated.'''
        return self._uncorrected

    def corrected(self):
        ''' Return the estimated x_corr .'''
        return (self._up+self._down)*0.5

    def correctedUp(self):
        ''' Return the estimated x_corr + up uncertainty.'''
        return self._up

    def correctedDown(self):
        ''' Return the estimated x_corr + down uncertainty.'''
        return self._down

    def nominal(self, x):
        ''' return corrected values for the x input using the nominal
        correction. approximating g with a univariate spline
        '''
        return (self._upInterp(x)+self._downInterp(x))*0.5

    def up(self, x):
        ''' return corrected values for the x input using the up
         correction, approximating g with a univariate spline
         '''
        return self._upInterp(x)

    def down(self, x):
        ''' return corrected values for the x input using the down
         correction,approximating g with a univariate spline
         '''
        return self._downInterp(x)

    def savetxt(self, name):
        ''' save the result in a txt (csv) file
            columns are
            input , nominal , down , up
            variations.
        '''
        result = np.column_stack((
            self._uncorrected, (self._down+self._up)*0.5,
            self._down, self._up))
        np.savetxt(name, result)
