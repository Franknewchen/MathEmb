import torch.utils.data
from batch import BatchMaskingAndSubstructContext


class DataLoaderMaskingAndSubstructContext(torch.utils.data.DataLoader):

    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMaskingAndSubstructContext, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMaskingAndSubstructContext.from_data_list(data_list),
            **kwargs)
