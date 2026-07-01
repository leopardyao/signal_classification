import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict


class Point:
    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx


class Pair:
    def __init__(self, p1, p2, p3, slope, intercept, intercept_p2, direction, num_outside, num_outside_p2, dist):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.slope = slope
        self.intercept = intercept
        self.intercept_p2 = intercept_p2
        self.direction = direction
        self.num_outside = num_outside
        self.num_outside_p2 = num_outside_p2
        self.dist = dist
        
    def print_info(self):
        print('2 upper/lower points:')
        print(self.p1.x, self.p1.y)
        print(self.p3.x, self.p3.y)
        print()
        print('1 middle point:')
        print(self.p2.x, self.p2.y)
        print()
        print('2 lines:')
        print(self.slope, self.intercept, self.intercept_p2, self.direction)
        print('num_outside: ', self.num_outside, self.num_outside_p2)
        print()
        print('dist: ', self.dist)
        print()


def get_pairs(array: np.array):
    results = []
    if array.ndim == 1:
        m = len(array)
        for i in range(m-2):
            for k in range(i+2, m):
                p1 = Point(i, array[i], i)
                p3 = Point(k, array[k], k)
                results.append([p1, p3])
    
    if array.ndim == 2:
        x = array[:, 0]
        y = array[:, 1]
        m = len(array)
        for i in range(m-2):
            for k in range(i+2, m):
                p1 = Point(x[i], y[i], i)
                p3 = Point(x[k], y[k], k)
                results.append([p1, p3])
    
    ## len n(n+1)/2
    return results


def search_lines(array, disable_tqdm=True): 
    array = np.asarray(array)
    pairs = get_pairs(array)
    if array.ndim == 1:
        x = np.arange(len(array))
        y = array[:]
    if array.ndim == 2:
        x = array[:, 0]
        y = array[:, 1]

    
    lines_dict = defaultdict(list)
    min_num_outside = 9999
    min_num_pairs = 9999    
    
    for p1, p3 in tqdm(pairs, disable=disable_tqdm):
        
        slope = (p1.y - p3.y) / (p1.x - p3.x)
        intercept = p1.y - slope * p1.x
        
        delta = intercept - y + slope * x
        #### 切线-测量值 <0 数量少, 线在上面, 平移后 >0 的要少, 线在下面
        ####            >0 数量少, 线在下面, 平移后 <0 的要少, 线在上面
        num_gt = np.sum(delta.astype(np.float16) > 0)
        num_lt = np.sum(delta.astype(np.float16) < 0)

        if num_gt < num_lt:
            num_outside = num_gt
            direction = 'lt' ## 平移后 <0 的要少，线在上面
        else:
            num_outside = num_lt
            direction = 'gt' ## 平移后 >0 的要少，线在下面
        
        if num_outside > min_num_pairs:
            continue
            
        min_num_pairs = num_outside

        
        rotate_values = [v - i * slope for i, v in zip(x, y)]
        if direction == 'lt':
            p2_idx = p1.idx + 1 + np.argmax(rotate_values[p1.idx+1:p3.idx])
            p2_x = x[p2_idx]
            p2_y = y[p2_idx]
         
            
            p1_y_r, p2_y_r, p3_y_r = rotate_values[p1.idx], rotate_values[p2_idx], rotate_values[p3.idx]
            if not (p2_y_r > p1_y_r and p2_y_r > p3_y_r):
                continue
            p2 = Point(p2_x, p2_y, p2_idx)
            intercept_p2 = p2.y - slope * p2.x
            delta_p2 = intercept_p2 - y + slope * x
            num_outside_p2 = np.sum(delta_p2.astype(np.float16) < 0)
            total_num_outside = num_outside + num_outside_p2
            
        if direction == 'gt':
            p2_idx = p1.idx + 1 + np.argmin(rotate_values[p1.idx+1:p3.idx])
            p2_x = x[p2_idx]
            p2_y = y[p2_idx]

            
            p1_y_r, p2_y_r, p3_y_r = rotate_values[p1.idx], rotate_values[p2_idx], rotate_values[p3.idx]
            if not (p2_y_r < p1_y_r and p2_y_r < p3_y_r):
                continue
            p2 = Point(p2_x, p2_y, p2_idx)
            intercept_p2 = p2.y - slope * p2.x
            delta_p2 = intercept_p2 - y + slope * x
            num_outside_p2 = np.sum(delta_p2.astype(np.float16) > 0)
            total_num_outside = num_outside + num_outside_p2
        

        if total_num_outside <= min_num_outside:
            dist = abs(intercept - intercept_p2) / np.sqrt(1 + slope ** 2)
            lines_dict[total_num_outside].append(Pair(p1, p2, p3, slope, intercept, intercept_p2, direction, num_outside, num_outside_p2, dist))
            min_num_outside = total_num_outside
        
    return lines_dict

def calculate_straightness_error(array):
    lines_dict = search_lines(array)
    
    ## find least distance
    min_num_outside = min(lines_dict.keys())
    lines = lines_dict[min_num_outside]
    min_dist = 9999
    result = None
    for each in lines:
        dist = each.dist
        if dist < min_dist:
            min_dist = dist
            result = each
            
    return result