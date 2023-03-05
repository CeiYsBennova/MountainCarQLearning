# MountainCarQLearning
Newbie RL, sometime good, always suck

#Set-up
> pip install gym or gymnasium
maybe need install pygame (Absolutely, maybe)

#Notice
If want to show pygame window and watch kiddo AI play the game, u need:
> env = gym.make('MountainCar-v0', render_mode="human")
but I do not encourage u to do it, it's a big mistake (time-consuming to render even with one episode). 