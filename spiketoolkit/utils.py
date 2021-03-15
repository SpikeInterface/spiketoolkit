import numpy as np

def get_closest_channels(recording, channel_ids, num_channels):
    closest_channels_id = []
    dist = []
    num_channels = min(num_channels, len(recording.get_channel_locations())-1)

    if not isinstance(channel_ids, list):
        channel_ids = list(channel_ids)

    for n, id in enumerate(channel_ids):
        locs = recording.get_channel_locations()
        distances = [np.linalg.norm(l - locs[id]) for l in locs]
        closest_channels_id.append(np.argsort(distances)[1:num_channels+1])
        dist.append(np.sort(distances)[1:num_channels+1])


    return np.array(closest_channels_id),np.array(dist)
