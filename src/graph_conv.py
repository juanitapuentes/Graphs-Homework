import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class SpatialGraphConv(MessagePassing):
    def __init__(self, coors, in_channels, out_channels, hidden_size, dropout=0):
        """
        coors - dimension of positional descriptors (e.g. 2 for 2D images)
        in_channels - number of the input channels (node features)
        out_channels - number of the output channels (node features)
        hidden_size - number of the inner convolutions
        dropout - dropout rate after the layer
        """
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.dropout = dropout
        self.lin_in = torch.nn.Linear(coors, hidden_size * in_channels)
        self.lin_out = torch.nn.Linear(hidden_size * in_channels, out_channels)
        self.in_channels = in_channels

    def forward(self, AAA, pos, BBB):

        """
        AAA - What do you expect to find in this matrix?
        pos - node position matrix [num_nodes, coors] 
        BBB - What do you expect to find in this matrix?
        """
        ###########################################################
        # TODO: Finish the parameters' description above 
        ########################################################

        BBB, _ = add_self_loops(BBB, num_nodes=AAA.size(0))   #num_edges = num_edges + num_nodes
        
        return self.propagate(edge_index=edge_index, x=AAA, pos=pos, aggr='add')  # [N, out_channels, label_dim]

    def message(self, pos_i, pos_j, x_j):
        """
        pos_i [num_edges, coors]
        pos_j [num_edges, coors]
        x_j [num_edges, label_dim]
        """
        
        ################# Put your code here ###################
        # TODO: Calculate the relative position using pos_j and pos_i
        ########################################################
        

        spatial_scaling = F.relu(self.lin_in(relative_pos))  # [n_edges, hidden_size * in_channels]
        
        
        ################# Put your code here ###################
        # TODO: Considering the previous steps, determine the current
        # number of nodes and assign it to the 'n_edges' variable.
        ########################################################
        
        
        result = spatial_scaling.reshape(n_edges, self.in_channels, -1) * x_j.unsqueeze(-1)
        return result.view(n_edges, -1)

    def update(self, aggr_out):
        """
        aggr_out [num_nodes, label_dim, out_channels]
        """
        aggr_out = self.lin_out(aggr_out)  # [num_nodes, label_dim, out_features]
        
        ################# Put your code here ###################
        # TODO: Add an activation function over aggr_out using the package torch_geometric.nn
        ########################################################
        
        aggr_out = F.dropout(aggr_out, p=self.dropout, training=self.training)

        return aggr_out
