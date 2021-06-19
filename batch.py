import torch
from torch_geometric.data import Data


class BatchMaskingAndSubstructContext(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchMaskingAndSubstructContext).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""

        batch = BatchMaskingAndSubstructContext()

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        for key in keys:
            batch[key] = []

        batch.batch = []
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0

        for data in data_list:
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long))
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                for key in data.keys:
                    item = data[key]
                    if key in ['edge_index', 'masked_atom_indices']:
                        item = item + cumsum_node
                    elif key == 'connected_edge_indices':
                        item = item + cumsum_edge
                    elif key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                        item = item + cumsum_substruct if batch.cumsum(key, item) else item
                    elif key in ["overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context"]:
                        item = item + cumsum_context if batch.cumsum(key, item) else item
                    batch[key].append(item)

                cumsum_node += num_nodes
                cumsum_edge += data.edge_index.shape[1]
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        """If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'face', 'masked_atom_indices', 'connected_edge_indices',"edge_index",
                       "edge_index_substruct", "edge_index_context", "overlap_context_substruct_idx",
                       "center_substruct_idx"]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
