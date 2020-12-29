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
         default ('data')

        - numBootstrap : Number of bootstrap samples (default 1000)

        - percInterpolation: interpolation option when calculating percentiles

        - smoothDegree : smoothDegree passed to
        scipy.interpolate.InterpolatedUnivariateSpline

        - Ext : Extrapolation option passed to
        scipy.interpolate.InterpolatedUnivariateSpline

      Notes:
      The grid of uncorrected values input,
      for which  a correction is estimated, is defined by
      input = percentiles(simul,target percentages)

      If bootstrapMode ='none' then
      correction = percentiles(data, target percentages)
      g =  InterpolatedUnivariateSpline( input, correction)
      and there are no uncertainties
      (the relevant methods return the same as the nominal)

      If bootstrapMode ='both'
      the method uses percentile smooth bootstrap
      to estimate the corrected values.
      Both data and simulation are resampled and a
      a correction^tilda for each value of the input
      is calculated via:
      g_resample = InterpolatedUnivariateSpline(simul_resample,data_resample )
      correction^tilda = g_resample(input).
      The correction^tilda from all data/MC bootstraps
      are used to derive the interval of correction^hat.
      The return values are:
      g_up = InterpolatedUnivariateSpline(input,correction^hat up)
      g_down = InterpolatedUnivariateSpline(input,correction^hat down)
      g_nominal = (g_up + g_down) * 0.5

      If bootstrapMode ='data' then the data are resampled
      and
      correction^tilda = percentiles(data_resample, target percentages)
      is calcualated.
      The correction^tilda from all data bootstraps
      are used to derive the interval of correction^hat.
      The return values are:
      g_up = InterpolatedUnivariateSpline(input,correction^hat up)
      g_down = InterpolatedUnivariateSpline(input,correction^hat down)
      g_nominal = (g_up + g_down) * 0.5
    '''

    def __init__(self,
                 data,
                 simul,
                 targetPerc,
                 bootstrapMode='data',
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
        # Array to keep track of the bootstrapss
        bootstrapResults = np.zeros(shape=(numBootstrap, numPercentiles))
        # Prepare for smooth bootstrap with normal Kernel
        normal = scipy.stats.norm()
        lenData = len(data)
        meanData = np.mean(data)
        varData = np.var(data)
        effSigmaData = min(math.sqrt(varData), scipy.stats.iqr(
            data)/1.34)
        hdata = 0.9 * effSigmaData * math.pow(lenData, -0.2)
        shrinkData = math.sqrt(1 + (hdata*hdata) * 1./varData)
        lenSimul = len(simul)
        meanSimul = np.mean(simul)
        varSimul = np.var(simul)
        effSigmaSim = min(math.sqrt(varSimul), scipy.stats.iqr(
            simul)/1.34)
        hsimul = 0.9 * effSigmaSim * math.pow(lenSimul, -0.2)
        shrinkSimul = math.sqrt(1 + (hsimul*hsimul) * 1./varSimul)

        # create bootstraps
        for i in range(numBootstrap):
            # resample the inputs with replacement with random noise
            epsData = normal.rvs(size=data.size)
            iData = np.random.choice(data, lenData, replace=True)
            iData = meanData + (iData - meanData + hdata*epsData)/shrinkData

            epsSimul = normal.rvs(size=simul.size)
            iSimul = np.random.choice(simul, lenSimul, replace=True)
            iSimul = meanSimul + (iSimul - meanSimul +
                                  hsimul*epsSimul)/shrinkSimul
            # create a mapping from the resamples
            percentileData = np.percentile(
                iData, q=targetPerc, interpolation=percInterpolation)
            percentileSim = np.percentile(
                iSimul, q=targetPerc, interpolation=percInterpolation)
            correctionRepl = scipy.interpolate.InterpolatedUnivariateSpline(
                percentileSim, percentileData, k=smoothDegree, ext=Ext)
            # call the resample mapping
            bootstrapResults[i] = correctionRepl(self._uncorrected)

        # Calculate up, down , nominal
        quant = np.array([2.5, 97.5])
        for i in range(numPercentiles):
            down, up = np.percentile(bootstrapResults[:, i], q=quant)
            self._down[i] = down
            self._up[i] = up

        # Create interpolated g(x_input)
        self._upInterp = scipy.interpolate.InterpolatedUnivariateSpline(
            self._uncorrected, self._up, k=smoothDegree, ext=Ext)
        self._downInterp = scipy.interpolate.InterpolatedUnivariateSpline(
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
        # Array to keep track of the bootstrapss
        bootstrapResults = np.zeros(shape=(numBootstrap, numPercentiles))
        # Prepare for smooth bootstrap with normal Kernel
        normal = scipy.stats.norm()
        lenData = len(data)
        meanData = np.mean(data)
        varData = np.var(data)
        effSigmaData = min(math.sqrt(varData), scipy.stats.iqr(
            data)/1.34)
        hdata = 0.9 * effSigmaData * math.pow(lenData, -0.2)
        shrinkData = math.sqrt(1 + (hdata*hdata) * 1./varData)
        # create bootstraps
        for i in range(numBootstrap):
            # resample the inputs with replacement with random noise
            epsData = normal.rvs(size=data.size)
            iData = np.random.choice(data, lenData, replace=True)
            iData = meanData + (iData - meanData + hdata*epsData)/shrinkData
            bootstrapResults[i] = np.percentile(
                iData, q=targetPerc, interpolation=percInterpolation)

        # down,nominal,up
        quant = np.array([2.5, 97.5])
        for i in range(numPercentiles):
            down, up = np.percentile(bootstrapResults[:, i], q=quant)
            self._down[i] = down
            self._up[i] = up

        # Create interpolated g(x_input)
        self._upInterp = scipy.interpolate.InterpolatedUnivariateSpline(
            self._uncorrected, self._up, k=smoothDegree, ext=Ext)
        self._downInterp = scipy.interpolate.InterpolatedUnivariateSpline(
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
        self._upInterp = scipy.interpolate.InterpolatedUnivariateSpline(
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
