import paramiko
import sys

host = "192.168.2.59"
user = "dev"
password = "Cesi_AI_cesI"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

stdin, stdout, stderr = client.exec_command("ps aux | grep docker")
print("--- PS OUTPUT ---")
print(stdout.read().decode('utf-8'))

client.close()
