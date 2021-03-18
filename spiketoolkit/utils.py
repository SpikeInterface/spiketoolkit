import numpy as np

def get_closest_channels(recording, channel_ids, num_channels=None):
    closest_channels_id = []
    dist = []

    if num_channels:
        num_channels = min(num_channels+1, len(recording.get_channel_locations()))
    else:
        num_channels = len(recording.get_channel_locations())

    if not isinstance(channel_ids, list):
        channel_ids = list(channel_ids)

    for n, id in enumerate(channel_ids):
        locs = recording.get_channel_locations()
        distances = [np.linalg.norm(l - locs[id]) for l in locs]
        closest_channels_id.append(np.argsort(distances)[1:num_channels])
        dist.append(np.sort(distances)[1:num_channels])


    return np.array(closest_channels_id),np.array(dist)
