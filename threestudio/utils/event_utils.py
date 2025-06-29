
import torch
import math
import random
import numpy as np
from itertools import product


def lin_log(x, threshold=20):
    """
    linear mapping + logarithmic mapping.
    :param x: float or ndarray the input linear value in range 0-255
    :param threshold: float threshold 0-255 the threshold for transisition from linear to log mapping
    """
    # converting x into np.float32.
    if x.dtype is not torch.float64:
        x = x.double()
    f = (1./threshold) * math.log(threshold)
    y = torch.where(x <= threshold, x*f, torch.log(x))

    return y.float()


def event_loss_call_window(
    all_rgb, 
    event_data,
    crop_box,
    combination
    ):
    '''
    simulate the generation of event stream and calculate the event loss
    '''

    rgb2grey = torch.tensor([0.299,0.587,0.114]).cuda()

    select_num=1
    random_idx = np.random.permutation(18)[:select_num]
    event_data = event_data[random_idx]
    chose = torch.tensor(combination)[random_idx]
    
    loss = []
    for its in range(select_num):
        start = chose[its][0]
        end = chose[its][1]

        rgb_start = all_rgb[start].reshape(-1, 3)
        rgb_end = all_rgb[end].reshape(-1, 3)
        
        thres_pos = (lin_log(torch.mv(rgb_end, rgb2grey) * 255) - lin_log(torch.mv(rgb_start, rgb2grey) * 255)) / 0.25
        thres_neg = (lin_log(torch.mv(rgb_end, rgb2grey) * 255) - lin_log(torch.mv(rgb_start, rgb2grey) * 255)) / 0.25
        
        bii = event_data[its][crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]].reshape(-1)

        pos = bii > 0
        neg = bii < 0

        loss_pos = torch.mean(((thres_pos * pos) - (bii * pos)) ** 2)
        loss_neg = torch.mean(((thres_neg * neg) - (bii * neg)) ** 2)

        loss.append(loss_pos + loss_neg)

    event_loss = torch.mean(torch.stack(loss, dim=0), dim=0)
    return event_loss


def interpolate_subpixel(x, y, v, w, h, image=None):
    image = image if image is not None else np.zeros((h, w), dtype=np.float32)

    if x.size == 0:
        return image

    # Implement the equation:
    # V(x,y) = \sum_i{ value * kb(x - xi) * kb(y - yi)}
    # We just consider the 4 integer coordinates around
    # each event coordinate, which will give a nonzero k_b()
    round_fns = (np.floor, np.ceil)

    k_b = lambda a: np.maximum(0, 1 - np.abs(a))
    xy_round_fns = product(round_fns, round_fns)
    for x_round, y_round in xy_round_fns:
        x_ref = x_round(x)
        y_ref = y_round(y)

        # Avoid summing the same contribution multiple times if the
        # pixel or time coordinate is already an integer. In that
        # case both floor and ceil provide the same ref. If it is an
        # integer, we only add it if the case #_round is torch.floor
        # We also remove any out of frame or bin coordinate due to ceil
        valid_ref = np.logical_and.reduce([
            np.logical_or(x_ref != x, x_round is np.floor),
            np.logical_or(y_ref != y, y_round is np.floor),
            x_ref < w, y_ref < h])
        x_ref = x_ref[valid_ref]
        y_ref = y_ref[valid_ref]

        if x_ref.shape[0] > 0:
            val = v[valid_ref] * k_b(x_ref - x[valid_ref]) * k_b(y_ref - y[valid_ref])
            np.add.at(image, (y_ref.astype(np.int64), x_ref.astype(np.int64)), val)

    return image


def brightness_increment_image(x, y, p, w, h, c_pos, c_neg, interpolate=True, threshold=False):
    assert c_pos is not None and c_neg is not None

    image_pos = np.zeros((h, w), dtype=np.float32)
    image_neg = np.zeros((h, w), dtype=np.float32)
    events_vals = np.ones([x.shape[0]], dtype=np.float32)

    pos_events = p > 0
    neg_events = np.logical_not(pos_events)

    if interpolate:
        image_pos = interpolate_subpixel(x[pos_events], y[pos_events], events_vals[pos_events], w, h, image_pos)
        image_neg = interpolate_subpixel(x[neg_events], y[neg_events], events_vals[neg_events], w, h, image_neg)
    else:
        np.add.at(image_pos, (y[pos_events].astype(np.int64), x[pos_events].astype(np.int64)), events_vals[pos_events])
        np.add.at(image_neg, (y[neg_events].astype(np.int64), x[neg_events].astype(np.int64)), events_vals[neg_events])
    
    if not threshold:        
        image = image_pos.astype(np.float32) - image_neg.astype(np.float32)
    else:
        image = image_pos.astype(np.float32) * c_pos - image_neg.astype(np.float32) * c_neg
    return image


def polarity_func(timestamp, events, id_to_coords):
    if timestamp < events[:, 1][0] + 5000:
        start_timestamp, end_timestamp = events[:, 1][0], events[:, 1][0] + 5000
    elif timestamp > events[:, 1][-1] -5000:
        start_timestamp, end_timestamp = events[:, 1][-1] - 5000, events[:, 1][-1]
    else:
        start_timestamp, end_timestamp = timestamp - 5000, timestamp + 5000

    event_start_idx = torch.searchsorted(events[:, 1], torch.tensor([start_timestamp]))
    event_end_idx = torch.searchsorted(events[:, 1], torch.tensor([end_timestamp]), side="right")

    event_data = events[event_start_idx:event_end_idx]

    x, y = id_to_coords[event_data[:, 0].long()].T.cpu().numpy()
    p = event_data[:, 2].cpu().numpy()

    bii = brightness_increment_image(x, y, p, 346, 260, 0.25, 0.25, interpolate=True)
    
    polarity = torch.abs(torch.from_numpy(bii))
    polarity = (polarity - polarity.min()) / (polarity.max() - polarity.min())
    polarity = ((polarity * 2) - 1)

    return polarity.sigmoid()