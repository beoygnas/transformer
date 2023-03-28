
def epoch_time(start_time, end_time) : 
    elapsed_time = end_time - start_time 
    elasped_mins = int(elapsed_time/60)
    elapsed_secs = int(elapsed_time - 60 * elasped_mins)
    return elasped_mins, elapsed_secs