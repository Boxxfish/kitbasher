from kitbasher.train import QNet, volume_fill_scorer
from kitbasher.env import NODE_DIM, ConstructionEnv
from torch_geometric.data import Batch # type: ignore

env = ConstructionEnv(volume_fill_scorer, use_potential=True, max_steps=64)
obs1 = env.reset()[0]
obs2 = env.step(10)[0]
batch = Batch.from_data_list([obs1, obs2])
net = QNet(3, NODE_DIM, 64)
out = net(batch)
print(out)