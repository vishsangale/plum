import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "runs"
# Find the latest event file
event_files = []
for root, dirs, files in os.walk(log_dir):
    for file in files:
        if "tfevents" in file:
            event_files.append(os.path.join(root, file))

if not event_files:
    print("No event files found.")
    exit()

latest_file = max(event_files, key=os.path.getmtime)
print(f"Reading {latest_file}...")

ea = EventAccumulator(latest_file)
ea.Reload()

tags = ea.Tags()['scalars']
if "SID/Unique_Codes" in tags:
    events = ea.Scalars("SID/Unique_Codes")
    if events:
        print(f"Latest Unique Codes: {events[-1].value} at step {events[-1].step}")
    else:
        print("Metric found but no events.")
else:
    print("SID/Unique_Codes not found in logs.")
    print("Available tags:", tags)
