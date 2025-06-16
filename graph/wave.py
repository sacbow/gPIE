from core.uncertain_array import UncertainArray

class Wave:
    def __init__(self):
        # Belief is initially None, will be set to an UncertainArray later
        self.belief = None

        # List of factors (e.g., Prior or Propagator objects), initially empty
        self.factors = []

        # List of input messages (UncertainArray), one per factor, initially empty
        self.inputs = []
