def log(instance, msg):
    with open(f"logs/log_{instance}.txt", "a") as f:
        f.write(msg + "\n")
