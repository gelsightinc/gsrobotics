import inspect
from typing import Callable

# List of subscribed event handlers
subscribers: list[Callable[[str], None]] = []


def log_message(message: str):
    """
    Logs a message and sends notification to subscribers.
    Automatically detects name of the script that calls log_message
    and adds it as a prefix to print statement.

    Args:
        message (str): message to log
    """

    caller = inspect.stack()[1].filename if len(inspect.stack()) > 1 else None
    script_name = caller.split("/")[-1].split("\\")[-1] if caller else ""
    log_output = f"{script_name.upper()} >> {message}"

    print(log_output)

    for subscriber in subscribers:
        subscriber(log_output)


def subscribe_to_logs(callback: Callable[[str], None]):
    """
    Allows external scripts to subscribe to the log messages,
    use them and process them.

    Args:
        callback (Callable[[str], None]): subscriber
    """

    if callback not in subscribers:
        subscribers.append(callback)


def unsubscribe_from_logs(callback: Callable[[str], None]):
    """
    Removes subscriber from event list.

    Args:
        callback (Callable[[str], None]): _description_
    """

    if callback in subscribers:
        subscribers.remove(callback)
