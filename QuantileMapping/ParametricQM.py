def parametricQM(input,
                 correction,
                 simulation):
    '''
    Applies  correction.ppf(simulation.cdf(input))
    Inputs
      - input to be corrected
      - correction object providing a ppf method
        e.g obj.ppf(values)
      - simulation object providing a cdf method
        e.g obj.cdf(values)
    '''
    return correction.ppf(simulation.cdf(input))
