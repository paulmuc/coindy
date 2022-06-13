from PyQt5.QtCore import QObject, pyqtSignal


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
