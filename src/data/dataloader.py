# src/data/dataloader.py
import torch
import queue
import threading
from torch.utils.data import DataLoader


class PrefetchLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.queue = queue.Queue(maxsize=2)
        self.worker = threading.Thread(target=self._worker)
        self.worker.daemon = True
        self.worker.start()

    def _worker(self):
        for batch in self.loader:
            self.queue.put(batch)
        self.queue.put(None)

    def __iter__(self):
        next_batch = self.queue.get()
        while next_batch is not None:
            with torch.cuda.stream(self.stream):
                next_batch = tuple(
                    t.to(self.device, non_blocking=True) for t in next_batch
                )
            torch.cuda.current_stream().wait_stream(self.stream)
            yield next_batch
            next_batch = self.queue.get()

    def __len__(self):
        return len(self.loader)


def create_dataloaders(dataset, training_config: dict):
    """
    Create a dataloader with the specified configuration.

    Args:
        dataset: The dataset to create a loader for
        training_config: Dictionary containing training configuration
    """
    # Extract parameters from config
    batch_size = training_config["batch_size"]
    num_workers = training_config.get("num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if torch.cuda.is_available():
        loader = PrefetchLoader(loader, torch.device("cuda"))

    return loader
