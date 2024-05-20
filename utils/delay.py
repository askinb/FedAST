import random
from scipy.stats import halfnorm
import numpy as np

class delay(object):
    def __init__(self, config, inv_speed =1):
        self.inv_speed = inv_speed
        self.delay_generate_list = []
        for i in range(len(config.models)):
            if config.models[i].delay.name == 'shiftedUniform':
                temp = shiftedUniform(config.models[i].delay.mean*inv_speed, config.models[i].delay.support)
            elif config.models[i].delay.name == 'halfNormal':
                temp = halfNormal(config.models[i].delay.mean*inv_speed)
            elif config.models[i].delay.name == 'shiftedExp':
                temp = shiftedExp(config.models[i].delay.exp_mean*inv_speed, config.models[i].delay.shift*inv_speed)
            self.delay_generate_list.append(temp)     
    def generate(self, modelIdx):
        return self.delay_generate_list[modelIdx]()
    
class shiftedUniform(object):
    def __init__(self, mean, support):
        self.mean = mean
        self.support = support
        
    def __call__(self):
        return self.mean - self.support/2 + self.support*random.random()
    
class halfNormal(object):
    
    def __init__(self, mean):
        
        self.mean = mean
        self.scale = mean*np.sqrt(np.pi/2)
    def __call__(self):
        return halfnorm.rvs(loc = 0, scale = self.scale)

class shiftedExp(object):
    def __init__(self, exp_mean, shift):
        self.shift = shift
        self.exp_mean = exp_mean
    def __call__(self):
        return np.random.exponential(scale=self.exp_mean) + self.shift 