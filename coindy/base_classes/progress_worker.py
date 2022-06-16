class ProgressWorker:
    """
    Base class for computing classes like SDEModel and SDESimulator
    """
    show_progress = False

    def check_progress(self, progress, increment, message=""):
        """ Updates the progress display of each instance of ProgressWorker

        :param progress: Progress as a float between 0-100
        :param increment: Increment to increase the progress by
        :param message: Optional message to write before the update
        :return: Updated progress
        """
        progress = progress + increment
        if self.__class__.show_progress:
            if not message == '':
                print(message, end="\n")
            progress_bar(progress, 100)
        return progress


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
