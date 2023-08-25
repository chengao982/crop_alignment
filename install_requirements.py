import subprocess
import time

def run_subprocess(arg):
    process = subprocess.Popen(arg, stdout=subprocess.PIPE)
    while True:
        output = process.stdout.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

start_time = time.time()

installation_args = ['pip3', 'install', '-r', 'requirements.txt']
run_subprocess(installation_args)
print('Requirements installed')