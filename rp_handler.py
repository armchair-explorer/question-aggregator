import runpod
import subprocess

def handler(event):
    # 1) Kick off your shell script as a subprocess
    #    Make sure run.sh is executable or call it via `bash`.
    proc = subprocess.Popen(
        ["bash", "run.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,           # line‐buffered
        universal_newlines=True
    )

    # 2) Read its output line by line and print—these prints go to your Pod logs.
    for line in proc.stdout:
        print(line.rstrip())

    # 3) Wait for it to finish, and get its exit code
    exit_code = proc.wait()
    if exit_code != 0:
        raise RuntimeError(f"run.sh failed (exit {exit_code})")

    # 4) Return a final result
    return {"status": "success", "exit_code": exit_code}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

