import pickle

def save_model(model, filepath):
    """Save a trained model to a specified filepath using pickle."""
    with open(filepath, 'wb') as file:
        pickle.dump(model, file)
def load_model(filepath):
    """Load a model from a specified filepath."""
    with open(filepath, 'rb') as file:
        model = pickle.load(file)
    return model
