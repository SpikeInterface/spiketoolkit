from subprocess import Popen, PIPE, CalledProcessError, call
import shlex


def run_command_and_print_output(command):
    with Popen(shlex.split(command), stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc


def call_command(command):
    try:
        call(shlex.split(command))
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)
