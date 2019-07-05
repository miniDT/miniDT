"""Module holds configuration of the setup relevant for reconstruction"""

class Config:
    """Holds information about the setup"""
    N_CHAMBERS = 4
    N_LAYERS = 4
    N_CELLS = 16
    X_CELL = 42.0
    Z_CELL = 13.0
    T_DRIFT = 390  # ns
    V_DRIFT = X_CELL * 0.5 / T_DRIFT

    def __init__(self):
        pass
