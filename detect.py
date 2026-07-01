import numpy as np
from scipy.signal import savgol_filter
import scipy
from pca_rotate import pca_rotation
from minzone import calculate_straightness_error

def detect(x, y, machine='Soutrac', threshold=100, method='pca', filter_window=5, filter_polyorder=3):
    '''
    (Preliminary screening)Classify group 1, 8 and 100 for gap position signal.

    input:
        x: x axis
        y: "pos" value
        machine: affect ptp threshold for group 8. If Soutrac, the threshold is 600. Otherwise, 250.
        threshold: ptp threshold for group 1
        method: "min_zone" for min zone method or "pca" for pca rotation

    return:
        group

    '''

    ## filter data first
    # y = savgol_filter(y, window_length=filter_window, polyorder=filter_polyorder)

    ## set default group
    group = 100

    if method == "pca":
        ptp = np.ptp(y)
        
        # 600 for soutrac, 250 for souspeed, soulas
        if machine in ['Soutrac']:
            group8_ptp_thres = 600
        else:
            group8_ptp_thres = 250

        if ptp < threshold:
            group = 1
        elif ptp > group8_ptp_thres: 
            group = 8
        else:
            rotated = pca_rotation(np.vstack((x,y)).T)
            pos_rotated = rotated[:, 1]
            ptp_rotated = np.ptp(pos_rotated)
            if ptp_rotated < threshold:
                group = 1
            else:
                group = 100

    if method == "min_zone":
        ptp = calculate_straightness_error(np.vstack((x,y)).T).dist
        if machine in ['Soutrac']:
            group8_ptp_thres = 600
        else:
            group8_ptp_thres = 250

        if ptp < threshold:
            group = 1
        elif ptp > group8_ptp_thres:
            group = 8
        else:
            group = 100
        
    return group


def get_consecutive_points(data: list):
    start = 0
    result = []
    
    for i in range(1, len(data)):
        if data[i] != data[i - 1] + 1:  # Check if the current number is not the next number in sequence.
            end = i
            seq = data[start:end]
            seq.append(seq[-1]+1)
            result.append(seq)  # Add the sequence to the result list.
            start = i  # Set the start of the next sequence as the current number.
    
    # Add the last sequence to the result list if it's not added inside the loop.
    seq = data[start:]
    if len(seq) > 0:
        seq.append(seq[-1]+1)
    result.append(seq)
    
    return result


def detect_group7(data, flat_mag=25, flat_avg=25, flat_len=10, extremities_diff=50,filter_window=5, filter_polyorder=3,verbose=False):
    '''
    Classify group 1, 7, or 100 for gap width when peak-to-peak difference of gapw is greater than 150.
    Pure function without any helpful data for drawing. Directly reuturn a group.

    input:
        data: (n,) array, "gapw"
        flat_mag: the threshold of peak-to-peak difference of a window for flat window detection
        flag_avg: the threshold of avg of a window for flat window detection
        filter_window: window_length in savgol filter
        verbose: if print info

    return:
        group
    '''

    ## init
    group = 7
    flat_st = None
    flat_ed = None
    up_st = None
    up_ed = None
    down_st = None
    down_ed = None
    lead = 1

    ## smooth signal
    smooth_signal = savgol_filter(data, window_length=filter_window, polyorder=filter_polyorder) # moving avg instead if high computation cost
    # smooth_signal = data ## without filter
    
    ## calculate derivative of signal
    derivative = np.diff(smooth_signal)

    ## statistical features of signal 
    mean = np.mean(derivative)
    std = np.std(derivative)
    med = np.median(derivative)
    # mode = scipy.stats.mode(derivative)
    mad = np.median(np.abs(derivative - np.median(derivative)))
    
    ## set threshold to separate up, down, flat trend
    increase_threshold = min(2, std)
    down_threshold = -increase_threshold
    flatline_threshold = increase_threshold

    ## get points of up/down/flat
    down_points = np.where(derivative < down_threshold)
    increase_points = np.where(derivative > increase_threshold)
    diff = np.abs(derivative)
    flatline_points = np.where(diff<flatline_threshold)

    ## number of points of up/down/flat
    n_down = len(down_points[0])
    n_up = len(increase_points[0])
    n_flat = len(flatline_points[0])

    ## extract consecutive points 
    up = get_consecutive_points(increase_points[0].tolist())
    flat = get_consecutive_points(flatline_points[0].tolist())
    down = get_consecutive_points(down_points[0].tolist())

    ## print info if verbose
    if verbose:
        print(f'derivative:\nmean: {mean}, std: {std}, median: {med}, mad: {mad}')
        print(f'down: {n_down}, up: {n_up}, flat: {n_flat}')
        print(f'down threshold: {down_threshold}')
        print(f'flat threshold: {flatline_threshold}')
        print(f'increase threshold: {increase_threshold}')

        print()
        print('up points:')
        print(up)
        print()
        print('down points:')
        print(down)
        print()
        print('flat points:')
        print(flat)

    ## if there is no flat trend, group 1
    if n_flat == 0:
        group = 1
        return group

    ## if no up or down, group 100
    if n_up == 0 and n_down == 0:
        group = 100
        return group
    ## else, determine which is main trend
    else:
        if n_up > n_down:
            lead = 1
        elif n_up < n_down:
            lead = -1
        else:
            ## tmp logic, should check the position of up/down based on flat but hard
            if np.ptp(data[increase_points[0]]) > np.ptp(data[down_points[0]]): 
                lead = 1
            else:
                lead = -1

    ## combine flat windows
    lfw = [] # longest flat window
    ## if up leads
    if lead == 1:
        for i in range(len(flat),0,-1):
            flat_seqs = flat[:i]    
            ## at least 1 flat window size should be > 3 
            max_len = max([len(each) for each in flat_seqs])
            if max_len < 4:
                break
            flat_st = flat_seqs[0][0]
            flat_ed = flat_seqs[-1][-1]
            seq_flat = smooth_signal[flat_st:flat_ed+1]
            ## determine if a potential flat window meets the criteria
            ## find the longest flat window
            if np.ptp(seq_flat) < flat_mag and np.mean(np.asarray(seq_flat)-np.min(seq_flat)) < flat_avg:
                lfw = seq_flat
                break
    ## if down leads
    if lead == -1:
        for i in range(len(flat)):
            flat_seqs = flat[i:]    
            ## at least 1 flat window size should be > 3 
            max_len = max([len(each) for each in flat_seqs])
            if max_len < 4:
                break
            flat_st = flat_seqs[0][0]
            flat_ed = flat_seqs[-1][-1]
            seq_flat = smooth_signal[flat_st:flat_ed+1]
            ## determine if a potential flat window meets the criteria
            ## find the longest flat window
            if np.ptp(seq_flat) < flat_mag and np.mean(np.asarray(seq_flat)-np.min(seq_flat)) < flat_avg:
                lfw = seq_flat
                break
    
    ## examine flat window
    ## if length of longest flat window < threshold, group 1
    if len(lfw) < flat_len:
        group = 1
        return group
    
    ## combine up/down windows
    if lead == 1:
        up_st = up[0][0]
        up_ed = up[-1][-1]
        if up_st < len(seq_flat):
            up_st = len(seq_flat)
    else:
        down_st = down[0][0]
        down_ed = down[-1][-1]
        if down_ed > flat_st:
            down_ed = flat_st

    ## examine extremities
    peak = max(smooth_signal) 
    if lead == 1:
        if up_ed < len(data) - 1:
            exts = smooth_signal[up_ed:]
            if peak - min(exts) > extremities_diff:
                group = 100
            else:
                up_ed = len(data) - 1
    
        if up_ed - up_st + 1 < 6:
            group = 100

        if flat_st > 0:
            exts = smooth_signal[:flat_st+1]
            if np.ptp(exts) > extremities_diff:
                group = 100
            else:
                flat_st = 0
    else:
        if down_st > 0:
            exts = smooth_signal[:down_st+1]
            if peak - min(exts) > extremities_diff: 
                group = 100
            else:
                down_st = 0

        if down_ed - down_st + 1 < 6:
            group = 100

        if flat_ed < len(data) - 1:
            exts = smooth_signal[flat_ed:]
            if np.ptp(exts) > extremities_diff:
                group = 100
            else:
                flat_ed = len(data) - 1
    
    return group


