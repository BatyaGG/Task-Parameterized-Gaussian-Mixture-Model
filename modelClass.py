class model:
    def __init__(self, nbStates, nbFrames, nbVar, ref, Priors, kP, kV, dt):
        self.nbStates = nbStates
        self.nbFrames = nbFrames
        self.nbVar = nbVar
        self.ref = ref
        self.Priors = Priors
        self.kP = kP
        self.kV = kV
        self.dt = dt