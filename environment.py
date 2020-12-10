import numpy as np
from class1 import PO as Ballon
from Target import Target
class Env:
    SIZE = 500
    RETURN_IMAGES = False
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 50
    #OBSERVATION_SPACE_VALUES = 8
    ACTION_SPACE_SIZE = 5

    def reset(self):
        self.player = Ballon("ballon", 50, 50, 1,1, 100, 1)
        self.food = Target(100,100,[100,100])
        self.food.instantiate()
        while self.food.check(self.player.x, self.player.y):
            self.food.instantiate()

        self.startdpos = [self.player.loc[0] - self.food.x, self.player.loc[1] - self.food.y]
        self.startdis = np.sqrt((self.food.x- self.player.loc[0])**2 + (self.food.y-self.player.loc[1])**2)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            self.dposold = [self.food.x- self.player.loc[0], self.food.y-self.player.loc[1]]
            observation = [self.dposold[0]/1920, self.dposold[1]/1080,  self.player.vel[0]/153, self.player.vel[1]/153, self.player.angvel/2.5, self.player.rot/(2*3.1415926535897932384623383)]

        return observation

    def step(self, action):
        #actie
        self.episode_step += 1
        self.player.action(action)

        #bewegen volgens natuurkunde simulatie
        self.player.setVelocity()
        self.player.move()
        #self.player.boundries(1920, 1080) Doe deze aan als je dat nodig vindt
        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            #delta x en delta y
            self.dpos = [self.food.x- self.player.loc[0], self.food.y-self.player.loc[1]]
            #de inputs worden geschaald zodat er een waarde uitkomt die ongeveer van nul tot 1 is, dat is beter voor het neural network.
            new_observation = [self.dposold[0]/1920, self.dposold[1]/1080,  self.player.vel[0]/153, self.player.vel[1]/153, self.player.angvel/2.5, self.player.rot/(2*3.1415926535897932384623383)]


        if self.food.check(self.player.loc[0], self.player.loc[1]):
            reward = self.FOOD_REWARD
        elif self.episode_step >= 2500:
            #reward aan het einde van de episode op basis van hoe erg de drone is genaderd.
            reward = 100 - (np.sqrt((self.food.x - self.player.loc[0])**2 + (self.food.y-self.player.loc[1])**2)/self.startdis)*100
        else:
            #reward voor elke step. -move_penalty zodat langer bezig zijn slechter is, de rest om in de juiste richting bewegen te belonen en in de verkeerde richting bewegen te bestraffen.
            xr = (np.abs(self.dposold[0]) - np.abs(self.dpos[0]))
            xr = xr if xr > 0 else 2*xr
            yr = (np.abs(self.dposold[1])-np.abs(self.dpos[1]))
            yr = yr if yr > 0 else 2*yr
            reward =  -self.MOVE_PENALTY + xr + yr

        done = False
        #kijken of we klaar zijn
        if reward == self.FOOD_REWARD or self.episode_step >= 2500:
            done = True
        #vervangen van de oude dpos
        self.dposold[0] = self.dpos[0]
        self.dposold[1] = self.dpos[1]

        return new_observation, reward, done

    def render(self, screen):
        self.player.draw(screen)
        self.food.draw(screen)
