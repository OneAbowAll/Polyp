import subprocess

def get_gpu_memory(message):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
        stdout=subprocess.PIPE,
        encoding='utf-8'
    )
    # Output: '1234, 8192\n'
    used, total = result.stdout.strip().split(', ')
    print(f"{message} GPU Memory Usage: {used} MB / {total} MB")
