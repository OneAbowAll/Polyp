DEBUG_LOG_ENABLE = True
WARNING_LOG_ENABLE = True
INFO_LOG_ENABLE = True

def print_debug(msg):
    if(DEBUG_LOG_ENABLE):
        print(msg)

def print_warning(msg):
    if(WARNING_LOG_ENABLE):
        print(msg)

def print_info(msg):
    if(INFO_LOG_ENABLE):
        print(msg)