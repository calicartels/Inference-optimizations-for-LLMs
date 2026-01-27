with open('scripts/train_distill.py', 'r') as f:
    lines = f.readlines()

# Insert parser creation after print_banner()
for i, line in enumerate(lines):
    if 'print_banner()' in line:
        lines.insert(i+1, '\n# Create argument parser\n')
        lines.insert(i+2, 'parser = argparse.ArgumentParser(description="Distillation training script")\n')
        lines.insert(i+3, '\n')
        break

with open('scripts/train_distill.py', 'w') as f:
    f.writelines(lines)

print("Fixed!")
