dfs_blk_size = 1433600  # * 1024

master_dir = "./master"
worker_dir = "./worker"

master_port = 4429  # DataNode程序监听端口
worker_port = 14429  # NameNode监听端口

# 集群中的主机列表
# host_list = ['thumm01', 'thumm02', 'thumm03', 'thumm04', 'thumm05', 'thumm06']  # ['thumm01', 'thumm02', 'thumm03', 'thumm04', 'thumm05', 'thumm06']
host_list = ['localhost']
master_host = "localhost"

BUF_SIZE = dfs_blk_size * 2
