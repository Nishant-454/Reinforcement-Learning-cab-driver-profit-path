# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():
   
    
    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =[]
        
        self.state_space = []
        for x in range(5):
            for y in range(x+1,5):
                self.action_space.append((x,y))
                self.action_space.append((y,x))
        self.action_space.append((0,0))  ### denoting the drver is not accepting the request
    
        for x in range(m):
            
            
            for y in range(0,t):
                
                for z in range(0,d):
                    self.state_space.append((x,y,z))
        
    
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()

    ####method to form action space

    ## Encoding state (or state-action) for NN input
    ### We wwill be using DQN where only state is passed

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        city = state[0]
        time =state[1]
        day = state[2]
        cityVector =np.zeros(m,dtype=int)
        timeVector =np.zeros(t,dtype=int)
        dayVector =np.zeros(d,dtype=int)
        
        cityVector[city] = 1
        timeVector[time] =1
        dayVector[day] = 1
        state_encod =list(cityVector)+list(timeVector)+list(dayVector)
        #state_encod = np.concatenate((cityVector,timeVector,dayVector),axis=0)
        #state_encod = state_encod.reshape( m + t + d) ### IF WE WANT VERTICAL ARRAY
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)            







        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        #actions = [self.action_space[i] for i in possible_actions_index]
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if(action == (0,0)):
            reward = -C
        else:
            driverCurrentLocation =state[0]
            timetoPickUpLocation=0
            if(driverCurrentLocation == action[0]):### if current location of driver and pick up location is same
                timetoPickUpLocation= 0
            else:
                timetoPickUpLocation = Time_matrix[driverCurrentLocation][action[0]][state[1]][state[2]]
            timetoDropLocation=Time_matrix[action[0]][action[1]][state[1]][state[2]]
            reward = R * timetoDropLocation - C*(timetoPickUpLocation + timetoDropLocation)
            
        return reward




    def next_state_func(self, state, action, Time_matrix,total_time):
        """Takes state and action as input and returns next state"""
        curr_loc = state[0]
        curr_time = state[1]
        curr_day = state[2]
        pickup_location = action[0]
        drop_location = action[1]
#        total_time = timetoDropLocation + waiting_time    ######total time is idle tie plus trip time
        terminal =False
        #####start the values
        
        #waiting_time=0
        timetoDropLocation = 0
        timetoPickUpLocation = 0
        #print("total time is",total_time)

    ######if pick up location and current location is same and and it is '0' then drive is uder wait time as time is integer we take it as '1' hr
        if pickup_location==0 and drop_location==0:
            total_time = total_time+1
            next_location=curr_loc
            curr_time = curr_time + 1
            
            
 ##if driver pickup location is current location and it get trip then next stete is drop location  
#####converting state means changing location, time and  day  
        else:
            if pickup_location == curr_loc:
                timetoDropLocation=Time_matrix[curr_loc][drop_location][curr_time][curr_day]
                
            else:
                timetoPickUpLocation=Time_matrix[curr_loc][pickup_location][curr_time][curr_day]
                ## adding timetoPickUpLocation because the current location of driver and picup_location  is different
                timetoDropLocation = Time_matrix[pickup_location][drop_location][curr_time][curr_day] + timetoPickUpLocation
            timetoDropLocation = int(timetoDropLocation)
            total_time = total_time+timetoDropLocation
            curr_time = curr_time + timetoDropLocation   
            next_location=drop_location   
            
        if(curr_time >23): 
            curr_time = curr_time - 24
            curr_day=curr_day+1
            if(curr_day > 6):  ### if it is the last day of week then we have to start the week again
                curr_day = 0
        next_state = [next_location, curr_time, curr_day]
        reward = self.reward_func( state, action, Time_matrix)
        #print("Reward is ",reward)
            #### checking for terminal
        if total_time>24*30: ###if hrs get completed it goes into charging mode and episode gets finished.
            terminal=True
        return next_state,reward,terminal,total_time




    def reset(self):
        return self.action_space, self.state_space, self.state_init
