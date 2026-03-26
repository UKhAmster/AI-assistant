import paramiko

host = "192.168.2.59"
user = "dev"
password = "Cesi_AI_cesI"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host, username=user, password=password)

sftp = client.open_sftp()
with sftp.file('AI-assistant/AI-assistant/Dockerfile', 'r') as f:
    text = f.read().decode('utf-8')

text = text.replace('pip install --no-cache-dir -r requirements.txt', 
                    'pip install --no-cache-dir --default-timeout=1000 -r requirements.txt')

with sftp.file('AI-assistant/AI-assistant/Dockerfile', 'w') as f:
    f.write(text.encode('utf-8'))

sftp.close()
client.close()
print("Successfully patched remote Dockerfile")
