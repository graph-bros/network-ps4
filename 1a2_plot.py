import networkx as nx
import matplotlib.pyplot as plt


# import graph
f = open("karate_club_edges.txt", "rb")
g = nx.read_edgelist(f)
f.close()

# partitions for social division and best partition
sd = [1,1,1,1,1,1,1,1,1,2,1,1,1,1,2,2,1,1,2,1,2,1,2,2,2,2,2,2,2,2,2,2,2,2]
bp = [2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2]

# 1. For social division
# apply partition labels into each vertex
mapping = {i: v for i, v in enumerate(sd)}
g_sd = nx.relabel_nodes(g, mapping)

# apply degree
d_sd = nx.degree(g_sd)
d_sd_size = []
for key in range(1, len(d_sd)+1):
    d_sd_size.append(100 * d_sd[str(key)])

nx.draw(g_sd, node_color=sd, node_size=d_sd_size)
plt.title('Social division', fontsize=20)
plt.savefig('social_division.png')
plt.gcf().clear()


# 2. For best partition
# apply partition labels into each vertex
mapping = {i: v for i, v in enumerate(bp)}
g_bp = nx.relabel_nodes(g, mapping)

# apply degree
d_bp = nx.degree(g_bp)
d_bp_size = []
for key in range(1, len(d_bp)+1):
    d_bp_size.append(100 * d_bp[str(key)])
nx.draw(g_bp, node_color=bp, node_size=d_bp_size)
plt.title('Best scoring partition', fontsize=20)
plt.savefig('best_scoring_partition.png')
plt.gcf().clear()
