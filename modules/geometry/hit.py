"""Module with classes for hits"""

class ProtoHit:
    """Hit as it is seen by the DAQ"""

    def __init__(self, chamber, layer, wire, time):
        self.chamber = chamber
        self.layer = layer
        self.wire = wire
        self.time = time


class Hit:
    """Hit as presented to the reconstruction algorithms"""

    x = 0.0

    def __init__(self, protoHit, side):
        self.protoHit = protoHit
        self.side = side

    @classmethod
    def fromProtoHit(cls, protoHit):
        """Returns a list of two hits (left + right) from a single ProtoHit"""
        return [cls(protoHit, -1), cls(protoHit, 1)]

