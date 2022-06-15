def progress_bar(progress: float, total: float):
    """ Prints a command line progress bar
    :param progress: Float indicating the progress with respect to total progress
    :param total: Float indicating the total progress
    """
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    if int(percent) == 100:
        print(f"\r|{bar}| {percent:.2f}%", end="\n\n")
    else:
        print(f"\r|{bar}| {percent:.2f}%", end="\r")
