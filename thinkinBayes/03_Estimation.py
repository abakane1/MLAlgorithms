
class Dice(suite):
    def likelihood(self, data, hype):
        if hype < data:
            return 0
        else:
            return 1.0/hype

suite = Dice([4, 6, 8, 12, 20])
