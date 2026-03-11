with open('src/main.py', 'r') as f:
    text = f.read()
text = text.replace('VAD_THRESHOLD = 0.5', 'VAD_THRESHOLD = 0.05')
with open('src/main.py', 'w') as f:
    f.write(text)
