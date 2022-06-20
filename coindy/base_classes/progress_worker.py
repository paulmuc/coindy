from coindy.utils.console_utils import progress_bar


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
