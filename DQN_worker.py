import gym
from DQN_RL import DQN
import socket
from common import*
import os
import numpy as np

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('MountainCar-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(
), int) else env.action_space.sample().shape     # to confirm the shape


class Worker:
    def __init__(self):
        self.dqn = DQN(type='worker')
        self.fc1_shape = self.dqn.eval_net.fc1.weight.data.shape
        self.out_shape = self.dqn.eval_net.out.weight.data.shape
        self.trans_flag = 0
        self.grads_flag = 0
        self.fc1_grads = np.empty((
            len(host_list), self.fc1_shape[0], self.fc1_shape[1]))
        self.out_grads = np.empty((
            len(host_list), self.out_shape[0], self.out_shape[1]))

    def run(self):  # 启动NameNode
        # 创建一个监听的socket
        listen_fd = socket.socket()
        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", worker_port))
            listen_fd.listen(5)
            print("Name node started")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("connected by {}".format(addr))
                print(sock_fd)

                try:
                    # 获取请求方发送的指令
                    request = self.recv_info(sock_fd)
                    print(request)
                    request = request.split(' ', 1)  # 指令之间使用空白符分割
                    print("Request: {}".format(request))

                    cmd = request[0]  # 指令第一个为指令类型
                    print(cmd)
                    if cmd == "models":  # 若指令类型为ls, 则返回DFS上对于文件、文件夹的内容

                        models = request[1]  # 指令第二个参数为DFS目标地址
                        print(request)
                        models = models.split('||')
                        print('models geshu', len(models))
                        self.send_trans(models)

                    elif cmd == "trans":  # 指令类型为获取FAT表项
                        trans = request[1]  # 指令第二个参数为DFS目标地址
                        self.send_grads(trans)

                    else:  # 其他位置指令
                        response = "Undefined command: " + " ".join(request)
                        print("Response: {}".format(response))

                except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
                    break
                except Exception as e:  # 如果出错则打印错误信息
                    print(e)
                finally:
                    sock_fd.close()  # 释放连接
        except KeyboardInterrupt:  # 如果运行时按Ctrl+C则退出程序
            pass
        except Exception as e:  # 如果出错则打印错误信息
            print(e)
        finally:
            listen_fd.close()  # 释放连接

    def send_trans(self, models):
        print(models)
        self.dqn.eval_net.fc1.weight.data = np.fromstring(
            eval(models[0])).reshape(self.fc1_shape[0], self.fc1_shape[1])
        self.dqn.eval_net.out.weight.data = np.fromstring(
            eval(models[1])).reshape(self.out_shape[0], self.out_shape[1])
        print("收到master传来的models,开始进行训练....", end='\t')
        self.dqn.rollout()
        print("训练结束..")
        master_sock = socket.socket()
        try:
            master_sock.connect(('localhost', master_port))
            request = 'trans'
            trans = self.dqn.memory
            request += ' ' + trans.tostring() + 'end'
            master_sock.send(bytes(request, encoding='utf-8'))
            check_info = str(master_sock.recv(BUF_SIZE), encoding='utf-8')
            print("已经传回获得的transition")
        except:
            check_info = 'dead'
        master_sock.close()

    def send_grads(self, memory):
        self.dqn.memory = np.fromstring(memory).reshape(-1, N_STATES*2+2)
        print("获得从master获得的memory", end='\t')
        self.dqn.compute_grad()
        fc1 = self.dqn.eval_net.fc1.weight.grad.data.tostring()
        out = self.dqn.eval_net.out.weight.grad.data.tostring()
        grads = fc1 + '|' + out
        master_sock = socket.socket()
        try:
            master_sock.connect(('localhost', master_port))
            request = 'grads'
            request += ' ' + grads.tostring() + 'end'
            master_sock.send(bytes(request, encoding='utf-8'))
            check_info = str(master_sock.recv(BUF_SIZE), encoding='utf-8')
            print("已经传回计算得到的grads..")
        except:
            check_info = 'dead'
        master_sock.close()

    def recv_info(self, sock_fd):
        # 从Client获取块数据
        result = ''
        # 接收的字符串
        acceptStr = ''
        # 循环接收客户端数据
        while True:
            # print("a")
            acceptStr = str(sock_fd.recv(BUF_SIZE), encoding='utf-8')
            print(acceptStr)
            # 将接受到的字符进行拼接
            result = result + acceptStr
            # 约定好客户端发送数据以'end'结束,这里可以和客户端进行约定以XXX结尾
            if acceptStr.endswith('end'):
                break
        # 处理拼接的字符，去掉'end'
        chunk_data = result.replace('end', '')

        return chunk_data


if __name__ == "__main__":
    worker_node = Worker()
    worker_node.run()
