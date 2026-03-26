import paramiko
import sys
import select

def run_remote_command(host, user, password, command):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(host, username=user, password=password)
        print(f"Executing: {command}")
        stdin, stdout, stderr = client.exec_command(command, get_pty=True)
        
        # If sudo is in command, send password
        if "sudo -S" in command:
            stdin.write(password + '\n')
            stdin.flush()
        
        # Read stdout and stderr continuously so it doesn't block the remote process
        while True:
            reads, _, _ = select.select([stdout.channel, stderr.channel], [], [], 1.0)
            if not reads and stdout.channel.exit_status_ready():
                break
            
            if stdout.channel.recv_ready():
                print(stdout.channel.recv(1024).decode('utf-8', errors='replace'), end='')
            if stderr.channel.recv_stderr_ready():
                print(stderr.channel.recv_stderr(1024).decode('utf-8', errors='replace'), end='', file=sys.stderr)
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    host = "192.168.2.59"
    user = "dev"
    password = "Cesi_AI_cesI"
    
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pwd && ls -la"
    run_remote_command(host, user, password, cmd)
