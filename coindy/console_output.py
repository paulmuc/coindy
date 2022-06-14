def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    if int(percent) == 100:
        print(f"\r|{bar}| {percent:.2f}%", end="\n\n")
    else:
        print(f"\r|{bar}| {percent:.2f}%", end="\r")
