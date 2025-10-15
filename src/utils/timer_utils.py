# src/utils/time_utils.py
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Union

class Timer:
    """ Stopwatch. Usage:

        timer = Timer()
        ... 
        print(timer.elapsed)         # seconds (float)
        print(timer.hms)             # (h, m, s)
        timer.display_timer()
        timer.stop_timer()
        timer.save_timer(dir_to_save)
    """
    def __init__(self):
        self._start = time.time()
        self._end = None

    def stop_timer(self):
        self._end = time.time()

    @property
    def elapsed(self) -> float:
        """ Compute elapsed time. """
        if self._end is None:
            return time.time() - self._start
        return self._end - self._start

    @property
    def hms(self):
        # self.time_diff = self._end - self._start
        # self.hours = int(self.time_diff // 3600)
        # self.minutes = int((self.time_diff % 3600) // 60)
        # self.seconds = self.time_diff % 60
        # self.time_diff_dict = {
        #     'hours': self.hours,
        #     'minutes': self.minutes,
        #     'seconds': round(self.seconds, 3)
        # }
        secs = int(self.elapsed)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return h, m, s

    def display_timer(self, print_fn=print):
        """ Display elapsed time in HH:MM:SS format. """
        h, m, s = self.hms
        print_fn(f"Elapsed Time: {h:02}:{m:02}:{s:02}")
    
    def get_elapsed_string(self) -> str:
        """ Return elapsed time as a formatted string (HH:MM:SS). """
        h, m, s = self.hms
        return f"{h:02}:{m:02}:{s:02}"

    def save_timer(self,
             dir_to_save: Union[str, Path] = '.',
             filename: str = 'runtime.json',
             extra: Optional[Dict] = None) -> None:
        """ Save runtime to JSON file. """
        hours, minutes, seconds = self.hms
        data = {'hours': hours, 'minutes': minutes, 'seconds': seconds}

        if extra is not None and not isinstance(extra, dict):
            print(f"Warning: 'extra' parameter is not a dictionary ({type(extra)}), ignoring.")
        elif isinstance(extra, dict):
            data.update(extra)

        dir_to_save = Path(dir_to_save)
        os.makedirs(dir_to_save, exist_ok=True)

        try:
            with open(dir_to_save / filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            return True
        except (IOError, TypeError) as e:
            print(f"Error saving timer: {e}")
            return False