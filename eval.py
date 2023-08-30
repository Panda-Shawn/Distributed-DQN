from DQN_RL import DQN
import gym


env = gym.make('Maze-v0')
env = env.unwrapped


model = DQN()
model.load_model('1349.pth')

# model.load_model('model_saved/eval.ptha, 14 Dec 2021 09:47:00 +0000.pth')

success_num = 0
total_num = 50
for i in range(total_num):
    s = env.reset()
    ep_r = 0
    done = False
    while not done:
        env.render()
        a = model.choose_action(s)
        print('state', s, '\taction', a)
        s_, r, done, info = env.step(a)
        if r == 5.0:
            success_num += 1
        ep_r += r
        s = s_

print("success rate:", success_num / total_num)
