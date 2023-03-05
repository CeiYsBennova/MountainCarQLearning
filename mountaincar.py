import gymnasium as gym
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

q_table_size = [20,20]
q_table_segment_size = (env.observation_space.high - env.observation_space.low)/q_table_size

def convert_standard_state(real_state):
    q_state = (real_state -env.observation_space.low) // q_table_segment_size
    return tuple(q_state.astype(np.int64))

#tao ma tran 3 chieu 20,20,3 the hien q table
q_table = np.random.uniform(low=-2, high=0, size=(q_table_size +[env.action_space.n]))


#setup Q-Learning
learning_rate = 0.1
discount = 0.95
eps = 1000
epsilon = 1
epsilon_min = 0.01
epsilon_decay_rate = 0.95

#tinh diem va hanh dong:
max_reward = -500
max_action_list = []
max_start_state = None

for ep in range(eps):
    termination = False
    current_state = convert_standard_state(env.reset()[0])
    ep_reward = 0
    action_list = []
    ep_start_state = current_state

    while not termination:
        #Lay gia tri max q-value cua current_state
        if np.random.random() > epsilon:
            action = np.argmax(q_table[current_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        action_list.append(action)

        #Thuc hien action theo gia tri tren
        new_real_state, reward, termination, _, _ = env.step(action=action)
        ep_reward+=reward

        if termination:
            if new_real_state[0] >= env.goal_position:
                print("Eps", ep, "done, reach reward = ", ep_reward)
                if ep_reward > max_reward:
                    max_reward = ep_reward
                    max_action_list = action_list
                    max_start_state = ep_start_state
        else:
            #convert ve standard
            new_state = convert_standard_state(new_real_state)
            
            #update Q value cho (current_state, action)
            current_q_value = q_table[current_state + (action,)]

            new_q_value = (1-learning_rate)*current_q_value + learning_rate*(reward+discount*np.max(q_table[new_state]))
        
            #update q value vao q_table
            q_table[current_state + (action,)] = new_q_value
            current_state = new_state
    if epsilon > epsilon_min:
        epsilon = epsilon*epsilon_decay_rate

print("Max reward = ", max_reward)
print("Action list for max reward: ", max_action_list)

env.reset()
env.state = max_start_state
for i in range(50):
    test_reward = 0
    for action in max_action_list:
        _,reward,_,_,_ = env.step(action)
        env.render()
        test_reward += reward
    
    print("Test reward = ", test_reward)