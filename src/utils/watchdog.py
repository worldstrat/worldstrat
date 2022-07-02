import json
import time
import os
import sys
import subprocess


def gpu_watchdog(minutes=5, gpu_usage_threshold=5):
    """ Monitors GPU usage and shuts down the computer/instance if the GPU usage 
    is less than the specified threshold for longer than the specified time.

    Parameters
    ----------
    minutes : int, optional
        The time in minutes the GPU usage needs to be below 5% in order to shut down the computer, by default 5.
    gpu_usage_threshold : int, optional
        The threshold in percentage the GPU usage needs to be below in order to shut down the computer, by default 5.
    """

    less_than_minutes = 0
    print(
        f"Watchdog running. If GPU usage is <={gpu_usage_threshold}% for {minutes} minutes, instance will be shut down."
    )
    while True:
        gpustat_json = subprocess.getoutput("gpustat --no-header --json")
        gpu_usage_percent = json.loads(gpustat_json)["gpus"][0]["utilization.gpu"]
        if gpu_usage_percent <= gpu_usage_threshold:
            less_than_minutes += 1
        else:
            less_than_minutes = 0
        # Less than minutes has been running for N minutes
        if less_than_minutes >= minutes * 60:
            print("GPU usage is less than 5% for {minutes} minutes. Stopping instance.")
            # Run the shutdown now command in shell
            os.system("sudo shutdown now")
        # Sleep for 1 second
        time.sleep(1)


if __name__ == "__main__":
    """ This is the main function that is called when the script is run. """
    if len(sys.argv) < 2 or sys.argv[1] == "--help" or sys.argv[1] == "-h":
        print(
            "Usage: python watchdog.py NUMBER_OF_MINUTES GPU_USAGE_THRESHOLD \n If the GPU is used less than GPU_USAGE_THRESHOLD for NUMBER_OF_MINUTES, the instance is shut down."
        )
    else:
        gpu_watchdog(int(sys.argv[1]), int(sys.argv[2]))
