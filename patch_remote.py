import paramiko

host = "192.168.2.59"
user = "dev"
password = "Cesi_AI_cesI"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

sftp = client.open_sftp()
with sftp.file('AI-assistant/AI-assistant/requirements.txt', 'r') as f:
    lines = f.readlines()

new_lines = ["--extra-index-url https://download.pytorch.org/whl/cu121\n"] + lines

with sftp.file('AI-assistant/AI-assistant/requirements.txt', 'w') as f:
    f.writelines(new_lines)

sftp.close()
client.close()
print("Successfully patched remote requirements.txt")
