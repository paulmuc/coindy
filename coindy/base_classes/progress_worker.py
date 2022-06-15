from PyQt5.QtCore import QObject, pyqtSignal

from coindy.utils.console_utils import progress_bar


class ProgressWorker(QObject):
    show_progress = False
    progress = pyqtSignal(float)
    finished = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self):
        super(ProgressWorker, self).__init__()
        self._cancel_requested = False

    def stop(self):
        self._cancel_requested = True

    def check_signals(self, index):
        self.progress.emit(index)
        if self._cancel_requested:
            self.stopped.emit()
            return False
        else:
            return True

    def check_progress(self, progress, increment, message=""):
        if not self.check_signals(progress):
            return
        progress = progress + increment
        if self.__class__.show_progress:
            if not message == '':
                print(message, end="\n")
            progress_bar(progress, 100)
        return progress
