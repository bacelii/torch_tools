def load_state(model,filepath:
    """
    Purpose: To load in a saved state dict into
    an instantiated model
    """
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)