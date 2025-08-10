class CombinedDataLoader:
    def __init__(self, data_loader, rel_loader):
        """
        Combine two dataloaders with the same number of batches.

        Parameters
        ----------
        data_loader : DataLoader
            The first dataloader containing data
        rel_loader : DataLoader
            The second dataloader containing relation matrix.
        """
        self.data_loader = data_loader
        self.rel_loader = rel_loader

    def __iter__(self):
        # Zip the two dataloaders
        return zip(iter(self.data_loader), iter(self.rel_loader))

    def __len__(self):
        # Ensure both dataloaders have the same number of batches
        return min(len(self.data_loader), len(self.rel_loader))

