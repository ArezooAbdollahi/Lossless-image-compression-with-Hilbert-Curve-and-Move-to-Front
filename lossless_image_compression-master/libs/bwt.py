#!/usr/bin/env python3

#
# Burrows Wheeler Transform implemented in Python
# Currently, list.sort() is not the radix sort, so the algorithm order is O(n log n log n).
# This algorithm replaces items of arr to integers, so we can use the radix sort instead.
# In that case, the algorithm order becomes O(n log n).
#
#collected from : https://gist.github.com/vbkaisetsu/d7c08e9c5aabe13686dd554ddfadf076
from operator import itemgetter

argsort = lambda l: [i for i, _ in sorted(enumerate(l), key=itemgetter(1))]


def suffix_array(arr):
    arr_size = len(arr)
    arr_int = {v: k for k, v in enumerate(sorted(set(arr)))}
    arr = [arr_int[x] for x in arr]
    arr.append(-1)
    suf = [[i, arr[i], arr[i + 1]] for i in range(arr_size)]
    suf.sort(key=itemgetter(1, 2))
    idx = [0] * arr_size
    k = 2
    while k < arr_size:
        r = 0
        prev_r = suf[0][1]
        for i in range(arr_size):
            if suf[i][1] != prev_r or suf[i - 1][2] != suf[i][2]:
                r += 1
            prev_r = suf[i][1]
            suf[i][1] = r
            idx[suf[i][0]] = i
        for i in range(arr_size):
            next_idx = suf[i][0] + k
            suf[i][2] = suf[idx[next_idx]][1] if next_idx < arr_size else -1
        suf.sort(key=itemgetter(1, 2))
        k <<= 1
    return [x[0] for x in suf]


def bwt(data):
    data_ref = suffix_array(data)
    return (x - 1 for x in data_ref), data_ref.index(0)


def ibwt(data, idx):
    sorted_data_ref = argsort(data)
    for i in range(len(data)):
        idx = sorted_data_ref[idx]
        yield idx

if __name__=='__main__':
    original = "Burrows Wheeler Transform"
    #original = "this is a test."
    #original = "abracadabra"
    bwt_ref, idx = bwt(original)
    encoded = "".join(original[x] for x in bwt_ref)
    print(encoded, idx)
    ibwt_ref = ibwt(encoded, idx)
    decoded = "".join(encoded[x] for x in ibwt_ref)
    print(decoded)
