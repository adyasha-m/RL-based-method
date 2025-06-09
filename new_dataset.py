import csv
import random

# --- Configuration ---
NUM_TARGETS = 10
NUM_TASKS = 100

# Radar capabilities: Which tasks each radar can perform
RADAR_CAPABILITIES = {
    'Radar1': ['Detection', 'Classification'],
    'Radar2': ['Classification', 'Tracking'],
    'Radar3': ['Tracking', 'Detection']
}
ALL_TASK_TYPES = ['Detection', 'Classification', 'Tracking']

# --- Column Generation Parameters ---
# Priority: 1 (highest) to 5 (lowest)
PRIORITY_RANGE = (1, 5)

# Start Time (Request Time): Simulate sequential arrivals
INITIAL_START_TIME = 0.0
# Average time between consecutive task requests
AVG_TIME_INCREMENT_BETWEEN_TASKS = 1.0

# Duration: Time units the task inherently takes
DURATION_RANGE = (2, 8)  # e.g., 2 to 8 time units

# Slack for Deadline: Additional time beyond (Start Time + Duration) for setting the deadline.
# End Time = Start Time + Duration + Slack.
# This slack also defines the Max Delay if the task runs for its nominal duration.
SLACK_FOR_DEADLINE_RANGE = (3, 10)  # e.g., 3 to 10 time units of slack

# Initial Power
INITIAL_POWER_RANGE = (10, 50)  # Power units

# Max Power: Must be >= Initial Power
# Max Power = Initial Power + Additional Power
ADDITIONAL_MAX_POWER_RANGE = (0, 20)  # Additional power units that can be provided

# --- Data Generation ---
tasks_data = []
current_simulation_time = INITIAL_START_TIME

for i in range(NUM_TASKS):
    task_id = i + 1
    # Assign tasks cyclically to targets (each target gets NUM_TASKS / NUM_TARGETS tasks)
    target_id = (i % NUM_TARGETS) + 1

    # 1. Task Type
    task_type = random.choice(ALL_TASK_TYPES)

    # 2. Radar Type (based on Task Type compatibility)
    eligible_radars = []
    for radar, capabilities in RADAR_CAPABILITIES.items():
        if task_type in capabilities:
            eligible_radars.append(radar)
    if not eligible_radars:
        # Fallback or error, though current config ensures this won't happen
        radar_type = "ErrorRadar"
    else:
        radar_type = random.choice(eligible_radars)

    # 3. Priority
    priority = random.randint(PRIORITY_RANGE[0], PRIORITY_RANGE[1])

    # 4. Start Time (Request Time)
    # Simulate tasks arriving over time with some variability
    time_increment = random.uniform(
        0.5 * AVG_TIME_INCREMENT_BETWEEN_TASKS,
        1.5 * AVG_TIME_INCREMENT_BETWEEN_TASKS
    )
    current_simulation_time += time_increment
    start_time = round(current_simulation_time, 2)

    # 5. Duration
    duration = random.randint(DURATION_RANGE[0], DURATION_RANGE[1])

    # 6. End Time (Deadline)
    # End Time = Start Time + Duration + Slack. Slack ensures task can be delayed or fit.
    slack_for_deadline = random.randint(SLACK_FOR_DEADLINE_RANGE[0], SLACK_FOR_DEADLINE_RANGE[1])
    end_time = round(start_time + duration + slack_for_deadline, 2)

    # 7. Initial Power
    initial_power = random.randint(INITIAL_POWER_RANGE[0], INITIAL_POWER_RANGE[1])

    # 8. Max Power
    # Max Power must be greater than or equal to Initial Power
    max_power = initial_power + random.randint(ADDITIONAL_MAX_POWER_RANGE[0], ADDITIONAL_MAX_POWER_RANGE[1])

    # 9. Max Delay
    # Max Delay = End Time - Start Time - Duration
    # This is the maximum time the task's execution can be postponed from its 'Start Time'
    # and still be completed by its 'End Time' if it runs for 'Duration'.
    # This is equivalent to the 'slack_for_deadline' used in End Time calculation.
    max_delay = round(end_time - start_time - duration, 2)
    # Ensure max_delay is not negative due to floating point arithmetic, though construction should prevent this.
    if max_delay < 0:
        max_delay = 0.0

    tasks_data.append([
        target_id,
        task_id,
        task_type,
        radar_type,
        priority,
        start_time,
        end_time,
        duration,
        initial_power,
        max_power,
        max_delay
    ])

# --- Write to CSV ---
csv_file_name = 'radar_tasks_dataset.csv'
headers = [
    'Target_ID', 'Task_ID', 'Task_Type', 'Radar_Type', 'Priority',
    'Request_Time', 'Deadline', 'Duration', 'Init_Power',
    'Max_Power', 'Max_Delay'
]

try:
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(tasks_data)
    print(f"CSV file '{csv_file_name}' generated successfully with {len(tasks_data)} tasks.")

    # Optional: Print first few rows for verification
    # print("\nFirst 5 tasks generated:")
    # for row in tasks_data[:5]:
    #     print(dict(zip(headers, row)))

except IOError:
    print(f"Error: Could not write to file '{csv_file_name}'. Please check permissions.")


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# df = pd.read_csv('./radar_tasks_dataset.csv')

# # Set aesthetic style for plots
# sns.set_style("whitegrid")

# # --- Visualizations ---

# # 1. Distribution of Task Types
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Task Type', palette='viridis')
# plt.title('Distribution of Task Types')
# plt.xlabel('Task Type')
# plt.ylabel('Number of Tasks')
# plt.show()

# # 2. Distribution of Radar Types
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Radar Type', palette='plasma')
# plt.title('Distribution of Radar Types')
# plt.xlabel('Radar Type')
# plt.ylabel('Number of Tasks')
# plt.show()

# # 3. Distribution of Priority
# plt.figure(figsize=(8, 5))
# sns.countplot(data=df, x='Priority', palette='magma')
# plt.title('Distribution of Task Priorities')
# plt.xlabel('Priority')
# plt.ylabel('Number of Tasks')
# plt.show()

# # 4. Histograms for numerical distributions
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# fig.suptitle('Distribution of Numerical Task Attributes', fontsize=16)

# sns.histplot(df['Start Time'], bins=20, kde=True, ax=axes[0, 0], color='skyblue')
# axes[0, 0].set_title('Distribution of Start Time')
# axes[0, 0].set_xlabel('Start Time (Time Units)')
# axes[0, 0].set_ylabel('Frequency')

# sns.histplot(df['Duration'], bins=10, kde=True, ax=axes[0, 1], color='lightcoral')
# axes[0, 1].set_title('Distribution of Duration')
# axes[0, 1].set_xlabel('Duration (Time Units)')
# axes[0, 1].set_ylabel('Frequency')

# sns.histplot(df['Max Delay'], bins=10, kde=True, ax=axes[0, 2], color='lightgreen')
# axes[0, 2].set_title('Distribution of Max Delay')
# axes[0, 2].set_xlabel('Max Delay (Time Units)')
# axes[0, 2].set_ylabel('Frequency')

# sns.histplot(df['Initial Power'], bins=10, kde=True, ax=axes[1, 0], color='cornflowerblue')
# axes[1, 0].set_title('Distribution of Initial Power')
# axes[1, 0].set_xlabel('Initial Power (Power Units)')
# axes[1, 0].set_ylabel('Frequency')

# sns.histplot(df['Max Power'], bins=10, kde=True, ax=axes[1, 1], color='orange')
# axes[1, 1].set_title('Distribution of Max Power')
# axes[1, 1].set_xlabel('Max Power (Power Units)')
# axes[1, 1].set_ylabel('Frequency')

# # End Time distribution
# sns.histplot(df['End Time'], bins=20, kde=True, ax=axes[1, 2], color='purple')
# axes[1, 2].set_title('Distribution of End Time')
# axes[1, 2].set_xlabel('End Time (Time Units)')
# axes[1, 2].set_ylabel('Frequency')


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# # 5. Relationship between Duration and Max Delay
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='Duration', y='Max Delay', hue='Task Type', size='Priority', palette='deep', sizes=(20, 200))
# plt.title('Duration vs. Max Delay (Colored by Task Type, Sized by Priority)')
# plt.xlabel('Duration (Time Units)')
# plt.ylabel('Max Delay (Time Units)')
# plt.legend(title='Task Type', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.show()

# # 6. Task Type vs. Duration
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x='Task Type', y='Duration', palette='coolwarm')
# plt.title('Task Type vs. Duration')
# plt.xlabel('Task Type')
# plt.ylabel('Duration (Time Units)')
# plt.show()

# # 7. Task Type vs. Initial Power
# plt.figure(figsize=(10, 6))
# sns.violinplot(data=df, x='Task Type', y='Initial Power', palette='viridis')
# plt.title('Task Type vs. Initial Power')
# plt.xlabel('Task Type')
# plt.ylabel('Initial Power (Power Units)')
# plt.show()

# # 8. Correlation Heatmap for numerical features
# plt.figure(figsize=(10, 8))
# numerical_cols = ['Priority', 'Start Time', 'End Time', 'Duration', 'Initial Power', 'Max Power', 'Max Delay']
# correlation_matrix = df[numerical_cols].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix of Numerical Attributes')
# plt.show()

# print("Visualizations generated successfully.")