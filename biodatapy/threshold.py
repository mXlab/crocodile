
class Threshold:
    def __init__(self, lower: float, upper: float):
        self.lower = lower
        self.upper = upper
        self.triggered = False

    def detect(self, value: float) -> bool:
        if ( value >= self.upper and self.triggered == False ):
            self.triggered = True
            return True
        elif ( value <= self.lower):
            self.triggered = False
        return False
