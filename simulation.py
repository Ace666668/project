import numpy as np


class MovingAgentSimulation:
    '''
    Simulation of COVID based on agents that can move around.
    This class solely focuses on simulation, and does NOTHING about
    visualization.

    A static method is included for mapping state values into colors.
    '''
    SUSCEPTIBLE = 1
    LATENT = 2
    INFECTED = 3
    RECOVER = 4

    def __init__(self, size, infect, symptom, recover, moving) -> None:
        '''
        Initialize simulation
        size: tuple of (nx, ny)
        infect: 0~1 ratio of probability that a susceptible turns into latent
        symptom: 0~1 ratio of probability that a latent turns into infected
        recover: 0~1 ratio of probability that an infected turns into recover
        moving: 0~1 ratio of probability that a site exchanges with its
                neighbor
        '''
        self.size = size
        self.infect = infect
        self.symptom = symptom
        self.recover = recover
        self.moving = moving
        # Initialize map with healthy people
        self.map = np.ones(size) * self.SUSCEPTIBLE
        # initialize step number
        self.n = 0

    def initialize(self, ratio, state=LATENT):
        '''
        Initialize state with random distribution of infected people.
        Ratio of infected people is given.
        '''
        rand = np.random.rand(self.size[0], self.size[1])
        self.map[rand < ratio] = state

    def step(self):
        '''
        Perform simulation step, use probabilities to update map, and return
        state after step
        '''
        # Calculate recover
        points = np.where(self.map == self.INFECTED)
        infects = list(zip(points[0], points[1]))
        recover_rate = np.random.rand(len(infects))
        for xy, r in zip(infects, recover_rate):
            if r <= self.recover:
                self.map[xy] = self.RECOVER

        # Calculate symptom
        points = np.where(self.map == self.LATENT)
        latents = list(zip(points[0], points[1]))
        symptom_rate = np.random.rand(len(latents))
        for xy, r in zip(latents, symptom_rate):
            if r <= self.symptom:
                self.map[xy] = self.INFECTED

        # Calculate infect
        points = np.where(self.map == self.SUSCEPTIBLE)
        susceptibles = list(zip(points[0], points[1]))
        infect_rate = np.random.rand(len(susceptibles))
        for xy, r in zip(susceptibles, infect_rate):
            if r <= self.infect:
                x, y = xy
                # Check if xy has an infected neighbor
                if (self.map[max(x-1, 0), y] > self.SUSCEPTIBLE or 
                    self.map[min(x+1, self.size[0] - 1), y] > self.SUSCEPTIBLE or 
                    self.map[x, max(y-1, 0)] > self.SUSCEPTIBLE or
                    self.map[x, min(y+1, self.size[1] - 1)] > self.SUSCEPTIBLE):
                    self.map[xy] = self.LATENT

        # Calculate exchange
        if self.moving > 0:
            self.exchange()

        # return state
        self.n += 1
        return self.n, self.map

    def exchange(self):
        '''
        Exchange people on the grid with given portion
        '''
        n = int(self.moving * self.size[0] * self.size[1])
        xs = np.random.randint(0, self.size[0], n)
        ys = np.random.randint(0, self.size[1], n)
        direction = np.random.randint(0, 4, n)
        for x, y, d in zip(xs, ys, direction):
            x2 = x
            y2 = y
            if d == 0:
                if x2 == 0:
                    continue
                x2 -= 1
            elif d == 1:
                if x2 == self.size[0] - 1:
                    continue
                x2 += 1
            elif d == 2:
                if y2 == 0:
                    continue
                y2 -= 1
            else:
                if y2 == self.size[1] - 1:
                    continue
                y2 += 1
            temp = self.map[x, y]
            self.map[x, y] = self.map[x2, y2]
            self.map[x2, y2] = temp

    def get_state(self):
        '''Get current state of simulation'''
        return (self.n, self.map)

    @staticmethod
    def convert_to_rgb(data):
        '''Convert simulation state to RGB info'''
        rgb = np.zeros((data.shape[0], data.shape[1], 3), int)
        rgb[data == MovingAgentSimulation.SUSCEPTIBLE] = (55, 55, 200)
        rgb[data == MovingAgentSimulation.LATENT] = (255, 255, 55)
        rgb[data == MovingAgentSimulation.INFECTED] = (200, 55, 55)
        rgb[data == MovingAgentSimulation.RECOVER] = (55, 200, 55)
        return rgb
