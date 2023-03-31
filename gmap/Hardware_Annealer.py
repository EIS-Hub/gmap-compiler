from simanneal import Annealer


class Hardware_Annealer(Annealer):
    def __init__(self, state):
        self.con = state
        super(Hardware_Annealer, self).__init__(state)
        self.con = self.state

    def move(self):
        self.con = self.state
        ret = self.update_cost()
        self.con = self.state
        return ret


    def energy(self):
        self.con = self.state
        ret = self.cost()
        self.con = self.state
        return ret

    def update_cost(self):
        pass

    def cost(self):
        pass

    def solve(self):
        self.con = self.state
        ret = self.anneal()
        self.con = self.state
        return ret
