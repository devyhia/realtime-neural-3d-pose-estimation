import numpy as np

def dataset_in_feature_space(session, model, dataset, dataset_list, batch_size=16):
    dataset_features = []
    for batch in dataset.batch_items(dataset_list, batch_size, shuffle=False):
        dataset_features.append(model(session, batch))
    
    return np.concatenate(dataset_features)