with open("jarvis.log", "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

target_lines = [l for l in lines if any(t in l for t in ["14:50:", "14:51:", "14:52:", "14:53:"])]
with open("scratch/filtered_log.txt", "w", encoding="utf-8") as out:
    out.writelines(target_lines)

print(f"Wrote {len(target_lines)} lines")

