# import pandas as pd
# import numpy as np

# # Set random seed for reproducibility
# np.random.seed(42)

# # Number of tasks
# n_tasks = 100

# # Generate each field
# task_ids = list(range(n_tasks))
# types = ['a', 'b', 'c']  # Dummy types
# radars = ['A', 'B', 'C']  # Dummy radar choices

# request_times = np.random.randint(0, 50, size=n_tasks)
# deadlines = request_times + np.random.randint(2, 6, size=n_tasks)  # Tight deadlines (2-5 units after request time)
# durations = np.random.randint(1, 6, size=n_tasks)  # Durations between 1 and 5
# init_powers = np.random.randint(1, 6, size=n_tasks)
# max_powers = init_powers + np.random.randint(1, 3, size=n_tasks)  # Max power 1-2 units more
# max_delays = np.round(np.random.uniform(0.5, 2.0, size=n_tasks), 2)  # Max delay allowed (optional field)

# # Randomly assign types and radars
# types_random = np.random.choice(types, size=n_tasks)
# radars_random = np.random.choice(radars, size=n_tasks)

# # Create DataFrame
# df = pd.DataFrame({
#     'Task_ID': task_ids,
#     'Type': types_random,
#     'Radar': radars_random,
#     'Request_Time': request_times,
#     'Deadline': deadlines,
#     'Duration': durations,
#     'Init_Power': init_powers,
#     'Max_Power': max_powers,
#     'Max_Delay': max_delays
# })

# # Save to CSV
# df.to_csv("hard_task_scheduling_dataset.csv", index=False)

# print("Dataset generated and saved to 'hard_task_scheduling_dataset.csv'")


import numpy as np
import pandas as pd

# Number of tasks
N = 100

# Define radar nodes and their capabilities
radars = {'A': ['a', 'b'], 'B': ['a', 'b', 'c', 'd'], 'C': ['c', 'd']}

np.random.seed(42)
tasks = []

for i in range(1,N+1):
    task_type = np.random.choice(['a', 'b', 'c', 'd'])  # Task type
    radar_options = [r for r, t in radars.items() if task_type in t]
    assigned_radar = np.random.choice(radar_options)  # Assign to a radar

    tnwb = np.random.randint(0, 50)  # Request time
    tnwe = tnwb + np.random.randint(2, 6)  # Deadline
    tnl = np.random.randint(1, 6)  # Duration
    Pn0 = np.random.randint(5, 10)  # Initial power
    Pnmax = Pn0 + np.random.randint(1, 3)  # Max power
    Dnmax =  np.round(np.random.uniform(0.5, 2.0), 2)  # Max delay
    
    tasks.append([i, task_type, assigned_radar, tnwb, tnwe, tnl, Pn0, Pnmax, Dnmax])

df = pd.DataFrame(tasks, columns=['Task_ID', 'Type', 'Radar', 'Request_Time', 'Deadline', 'Duration', 'Init_Power', 'Max_Power', 'Max_Delay'])

df.to_csv("dataset.csv", index=False)
print("Dataset generated and saved!")