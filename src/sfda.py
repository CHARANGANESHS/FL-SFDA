class BaseSFDA:
    def __init__(self, source_model, target_model):
        self.source_model = source_model
        self.target_model = target_model

    def adapt(self, target_loader, device):
        raise NotImplementedError
