def get_pair_permutation(num_items: int, n: int):
    running_count = num_items - 1
    running_n = n
    val_0 = 0
    while running_n > running_count:
        running_n -= running_count
        running_count -= 1
        val_0 += 1

    return val_0, val_0 + running_n
