class Creatdata():
    def __init__(self, data,targets) -> None:
        self.data = data
        self.targets = targets
        self.item_count = len(targets)
    
    def __len__(self):
        return self.item_count