import matplotlib.pyplot as plt
import re

# โหลดข้อมูลจากไฟล์ log.txt
with open("log.txt", "r") as file:
    logs = file.readlines()

# ใช้ regex เพื่อดึงค่า Epoch, D loss และ G loss
epochs = []
d_losses = []
g_losses = []

for line in logs:
    match = re.search(r"\[Epoch (\d+)/\d+\] \[D loss: ([\d\.-]+)\] \[G loss: ([\d\.-]+)\]", line)
    if match:
        epochs.append(int(match.group(1)))
        d_losses.append(float(match.group(2)))
        g_losses.append(float(match.group(3)))

# พล็อตกราฟ
plt.figure(figsize=(10, 5))
plt.plot(epochs, d_losses, label="D Loss", color="red")
plt.plot(epochs, g_losses, label="G Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("D Loss vs G Loss Over Epochs")
plt.legend()
plt.show()
