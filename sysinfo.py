import psutil

def sys_info():
    cpu_usage = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    ram_total = mem.total / 1e9
    ram_used = mem.used / 1e9

    return cpu_usage, ram_used