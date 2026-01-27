DEBUG_LOG_ENABLE = True
WARNING_LOG_ENABLE = True
INFO_LOG_ENABLE = True

def print_debug(msg):
    global DEBUG_LOG_ENABLE
    if DEBUG_LOG_ENABLE:
        print(msg)

def print_warning(msg):
    global WARNING_LOG_ENABLE
    if WARNING_LOG_ENABLE:
        print(msg)

def print_info(msg):
    global INFO_LOG_ENABLE
    if INFO_LOG_ENABLE:
        print(msg)