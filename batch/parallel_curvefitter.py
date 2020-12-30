from refnx.analysis import CurveFitter


def fit_an_objective(objective):
    # make the curvefitter and do the fit
    fitter = CurveFitter(objective)
    fitter.fit('differential_evolution', verbose=False, tol=0.05)
    return objective
