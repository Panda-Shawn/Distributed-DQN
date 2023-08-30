# 问题1，传回梯度的更新问题 直接net.weight.grad.data 直接赋值即可(已验证)
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


class Master:
    def __init__(self):
        self.dqn = DQN()
        self.fc1_shape = self.dqn.eval_net.fc1.weight.data.shape
        self.out_shape = self.dqn.eval_net.out.weight.data.shape
        self.trans_flag = 0
        self.grads_flag = 0
        self.fc1_grads = np.empty((
            len(host_list), self.fc1_shape[0], self.fc1_shape[1]))
        self.out_grads = np.empty((
            len(host_list), self.out_shape[0], self.out_shape[1]))

    def run(self):  # 启动NameNode
        self.distribute_models()
        # 创建一个监听的socket
        listen_fd = socket.socket()
        try:
            # 监听端口
            listen_fd.bind(("0.0.0.0", master_port))
            listen_fd.listen(5)
            print("Name node started")
            while True:
                # 等待连接，连接后返回通信用的套接字
                sock_fd, addr = listen_fd.accept()
                print("connected by {}".format(addr))

                try:
                    # 获取请求方发送的指令
                    request = self.recv_info(sock_fd)
                    request = request.split(' ', 1)  # 指令之间使用空白符分割
                    print("Request: {}".format(request))

                    cmd = request[0]  # 指令第一个为指令类型

                    if cmd == "trans":  # 若指令类型为ls, 则返回DFS上对于文件、文件夹的内容
                        trans = request[1]  # 指令第二个参数为DFS目标地址
                        trans = np.fromstring(trans).reshape(-1, N_STATES*2+2)
                        for tran in trans:
                            self.dqn.store_transition(
                                tran[:N_STATES], tran[N_STATES], tran[N_STATES+1], tran[N_STATES+2:])
                        self.trans_flag += 1

                    elif cmd == "grads":  # 指令类型为获取FAT表项
                        grads = request[1]  # 指令第二个参数为DFS目标地址
                        grads = grads.split('|')
                        self.fc1_grads[self.grads_flag] = np.fromstring(
                            grads[0]).reshape(self.fc1_shape[0], self.fc1_shape[1])
                        self.out_grads[self.grads_flag] = np.fromstring(
                            grads[1]).reshape(self.out_shape[0], self.out_shape[1])
                        self.grads_flag += 1

                    else:  # 其他位置指令
                        response = "Undefined command: " + " ".join(request)
                        print("Response: {}".format(response))

                    if self.trans_flag == len(host_list):
                        self.distribute_trans()
                        self.trans_flag = 0

                    if self.grads_flag == len(host_list):
                        fc1, out = self.compute_grads()
                        self.dqn.update_gradient(fc1, out)
                        self.dqn.save_model()
                        self.distribute_models()
                        self.grads_flag = 0

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

    def start_workers(self):
        for worker in host_list:
            work_sock = socket.socket()
            try:
                work_sock.connect((worker, worker_port))
                request = 'models'
                work_sock.send(bytes(request, encoding='utf-8'))
                check_info = str(work_sock.recv(BUF_SIZE), encoding='utf-8')
            except:
                check_info = 'dead'
            work_sock.close()

    def stop_workers(self):
        for worker in host_list:
            worker_sock = socket.socket()
            try:
                worker_sock.connect((worker, worker_port))
                request = 'stop'
                worker_sock.send(bytes(request, encoding='utf-8'))
                check_info = str(worker_sock.recv(BUF_SIZE), encoding='utf-8')
            except:
                check_info = 'dead'
            worker_sock.close()

    def distribute_trans(self):
        print("开始向所有的worker分发transition...")
        for worker in host_list:
            worker_sock = socket.socket()
            try:
                worker_sock.connect((worker, worker_port))
                request = 'trans'
                trans = self.dqn.sample_transition()
                request += ' ' + trans.tostring() + 'end'
                worker_sock.send(bytes(request, encoding='utf-8'))
                check_info = str(worker_sock.recv(BUF_SIZE), encoding='utf-8')
            except:
                check_info = 'dead'
            worker_sock.close()

    def distribute_models(self):
        print("开始向所有的worker分发参数...")

        # fc1, out = self.dqn.extract_model()
        # # fc1 = fc1.numpy()
        # # out = out.numpy()
        # print(type(fc1.numpy().tobytes()))
        # print("**")
        # request = 'models ' + f'{fc1.numpy().tostring()}' + '|' + \
        #     f'{out.numpy().tostring()}' + 'end'
        # print(request)

        for worker in host_list:
            worker_sock = socket.socket()
            try:
                worker_sock.connect((worker, worker_port))
                fc1, out = self.dqn.extract_model()
                request = 'models ' + f'{fc1.numpy().tostring()}' + '||' + \
                    f'{out.numpy().tostring()}' + 'end'
                print(request)
                worker_sock.send(bytes(request, encoding='utf-8'))
                # worker_sock.send(request)
                check_info = str(worker_sock.recv(BUF_SIZE), encoding='utf-8')
            except:
                check_info = 'dead'
            worker_sock.close()

    def compute_grads(self):
        return self.fc1_grads.mean(), self.out_grads.mean()

    def recv_info(self, sock_fd):
        # 从Client获取块数据
        result = ''
        # 接收的字符串
        acceptStr = ''
        # 循环接收客户端数据
        while True:
            # print('a')
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
    master_node = Master()
    master_node.run()
