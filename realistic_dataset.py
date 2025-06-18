import csv
import random

# --- Configuration for Realistic Task Generation ---

NUM_UNIQUE_TARGETS = 50 # Number of distinct targets to simulate
MAX_STAGES_PER_TARGET_TYPE = {
    'Search': 1,        # 1 Search task is enough to potentially move to Detection
    'Detection': 4,     # Need 4 Detection tasks to move to Tracking
    'Tracking': 5,      # Need 5 Tracking tasks to move to Classification
    'Classification': 6 # Need 6 Classification tasks for 'Target Locked'
}
STAGE_ORDER = ['Search', 'Detection', 'Tracking', 'Classification', 'Locked']

RADAR_CAPABILITIES = {
    'Radar1': ['Search', 'Detection', 'Classification'],
    'Radar2': ['Classification', 'Tracking', 'Detection'],
    'Radar3': ['Tracking', 'Search', 'Detection']
}

PRIORITY_MAP = {
    'Search': 1,
    'Detection': 2,
    'Tracking': 3,
    'Classification': 4
}
# DEFAULT_PRIORITY_RANGE is no longer strictly needed as priority is mapped to task type
# DEFAULT_PRIORITY_RANGE = (1, 5)

INITIAL_REQUEST_TIME = 0.0
# Average time between initial target appearances, leading to bursts
AVG_TARGET_ARRIVAL_INTERVAL = 5.0
MAX_SIMULATION_TIME = 1500.0 # Define a maximum time for task generation

# Realistic Task Parameter Configurations based on Justification:

# 1. Deadline Offsets: How much time (buffer) is typically allowed after request_time
#    Classification tasks have tighter deadlines due to urgency. Search tasks more lenient.
DEADLINE_OFFSETS_BY_TASK_TYPE = {
    'Search': (20.0, 50.0),        # 20 to 50 time units after request
    'Detection': (15.0, 40.0),
    'Tracking': (10.0, 30.0),
    'Classification': (5.0, 15.0)  # Tighter for critical classification
}

# 2. Duration Factors: Base duration for each task type.
#    More complex tasks (Tracking, Classification) take longer.
BASE_DURATION_BY_TASK_TYPE = {
    'Search': 3.0,
    'Detection': 5.0,
    'Tracking': 8.0,
    'Classification': 10.0
}
DURATION_VARIATION_FACTOR = 0.3 # +/- 30% of base duration

# 3. Power Requirements: Initial and Max power related to task complexity.
#    Higher task types generally need more power. Max_Power is a multiple of Init_Power.
BASE_INIT_POWER_BY_TASK_TYPE = {
    'Search': 20,
    'Detection': 40,
    'Tracking': 60,
    'Classification': 80
}
# Max_Power will be Init_Power * POWER_BOOST_FACTOR
POWER_BOOST_FACTOR_RANGE = (1.5, 2.5) # Max power is 1.5x to 2.5x initial power

# 4. Max Delay: Directly proportional to priority (inverse relationship: higher priority means less delay)
#    Multiplier for how much 'flexibility' a task has beyond its minimum required time.
#    Lower priority tasks can wait longer.
MAX_DELAY_MULTIPLIER_BY_PRIORITY = {
    4: (0.1, 0.5), # Priority 4 (Classification) can only delay by 10-50% of its duration
    3: (0.5, 1.0), # Priority 3 (Tracking) can delay by 50-100% of its duration
    2: (1.0, 2.0), # Priority 2 (Detection) can delay by 100-200% of its duration
    1: (2.0, 4.0)  # Priority 1 (Search) can delay by 200-400% of its duration
}


# --- Data Structures ---
tasks_data = []
task_id_counter = 0
# Stores the current state for each target: last request time, and stage counts
target_states = {i: {'last_request_time': INITIAL_REQUEST_TIME, 'stage_counts': {}} for i in range(NUM_UNIQUE_TARGETS)}
# Initialize stage counts for all targets to 0 for each stage type
for target_id in target_states:
    for stage in STAGE_ORDER:
        target_states[target_id]['stage_counts'][stage] = 0

# For tracking 'must-drop' tasks (challenging tasks designed to be difficult to complete)
must_drop_tasks_generated = 0
MUST_DROP_CHANCE = 0.05 # 5% chance for a task to be a 'must-drop' candidate
MUST_DROP_DEADLINE_FACTOR = 0.3 # Must-drop tasks have a deadline 30% of normal
MUST_DROP_POWER_FACTOR = 1.5 # Must-drop tasks require 50% more power
MUST_DROP_DURATION_FACTOR = 1.2 # Must-drop tasks are 20% longer
MUST_DROP_PRIORITY = 5 # Highest priority for must-drop tasks


# --- Task Generation Logic ---
# Keep generating tasks until MAX_SIMULATION_TIME is reached for all targets
active_targets = list(range(NUM_UNIQUE_TARGETS))
while active_targets:
    target_id = random.choice(active_targets)
    target_state = target_states[target_id]

    # Advance request_time for the chosen target
    # Simulate a more natural progression of new task requests
    target_state['last_request_time'] += random.uniform(0.5, 2.0) # Small increment for new tasks for the same target

    # Determine the next task type based on progression rules
    current_stage = 'Search'
    for i in range(len(STAGE_ORDER) - 1):
        stage = STAGE_ORDER[i]
        next_stage = STAGE_ORDER[i+1]
        required_count = MAX_STAGES_PER_TARGET_TYPE.get(stage, 1) # Default 1 if not specified
        if target_state['stage_counts'].get(stage, 0) < required_count:
            current_stage = stage
            break
        elif stage == 'Classification' and target_state['stage_counts'].get(stage, 0) >= required_count:
            current_stage = 'Locked' # Target is fully processed
            break
    next_task_type = current_stage

    if next_task_type == 'Locked':
        active_targets.remove(target_id)
        continue # This target is done, skip to the next active one

    request_time = target_state['last_request_time']

    # Stop generating tasks if we've exceeded the simulation time
    if request_time >= MAX_SIMULATION_TIME:
        active_targets.remove(target_id)
        continue

    # Choose a radar capable of performing this task type
    capable_radars = [r_type for r_type, caps in RADAR_CAPABILITIES.items() if next_task_type in caps]
    if not capable_radars:
        continue # Should not happen if RADAR_CAPABILITIES are well-defined
    chosen_radar_type = random.choice(capable_radars)

    # Determine Priority based on Task_Type
    priority = PRIORITY_MAP.get(next_task_type, 1) # Default to 1 if not mapped

    # --- Apply Realistic Constraints ---

    # 1. Deadline
    min_offset, max_offset = DEADLINE_OFFSETS_BY_TASK_TYPE.get(next_task_type, (10.0, 60.0))
    deadline = request_time + random.uniform(min_offset, max_offset)

    # 2. Duration
    base_duration = BASE_DURATION_BY_TASK_TYPE.get(next_task_type, 5.0)
    duration_variation = base_duration * DURATION_VARIATION_FACTOR
    duration = random.uniform(base_duration - duration_variation, base_duration + duration_variation)
    duration = max(0.1, duration) # Ensure duration is not negative or too small

    # 3. Power
    base_init_power = BASE_INIT_POWER_BY_TASK_TYPE.get(next_task_type, 50)
    init_power = random.uniform(base_init_power * 0.8, base_init_power * 1.2) # Small variation around base
    power_boost_factor = random.uniform(POWER_BOOST_FACTOR_RANGE[0], POWER_BOOST_FACTOR_RANGE[1])
    max_power = init_power * power_boost_factor
    max_power = max(init_power + 1, max_power) # Ensure max_power is always greater than init_power

    # 4. Max Delay
    min_mult, max_mult = MAX_DELAY_MULTIPLIER_BY_PRIORITY.get(priority, (0.5, 1.5))
    max_delay = duration * random.uniform(min_mult, max_mult)
    max_delay = max(0.0, max_delay) # Ensure delay is not negative

    # --- Introduce 'Must-Drop' Tasks with stricter constraints ---
    is_must_drop = False
    if random.random() < MUST_DROP_CHANCE and next_task_type in ['Tracking', 'Classification']:
        is_must_drop = True
        must_drop_tasks_generated += 1
        priority = MUST_DROP_PRIORITY # Force highest priority
        # Make deadline very tight
        deadline = request_time + duration * MUST_DROP_DEADLINE_FACTOR
        # Increase power requirements
        init_power *= MUST_DROP_POWER_FACTOR
        max_power *= MUST_DROP_POWER_FACTOR
        # Increase duration slightly (to make it harder)
        duration *= MUST_DROP_DURATION_FACTOR
        # Drastically reduce max delay
        max_delay = duration * 0.1 # Very little room for delay


    task_id_counter += 1
    target_state['stage_counts'][next_task_type] = target_state['stage_counts'].get(next_task_type, 0) + 1

    tasks_data.append([
        target_id, task_id_counter, next_task_type, chosen_radar_type, priority,
        request_time, deadline, duration, init_power, max_power, max_delay,
        target_state['stage_counts'][next_task_type]
    ])

    # Update the last request time for this target
    target_state['last_request_time'] = request_time

# --- Write to CSV ---
csv_file_name = 'realistic_dataset.csv' # Changed output filename
headers = [
    'Target_ID', 'Task_ID', 'Task_Type', 'Radar_Type', 'Priority',
    'Request_Time', 'Deadline', 'Duration', 'Init_Power',
    'Max_Power', 'Max_Delay', 'Stage_Count'
]

try:
    with open(csv_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(tasks_data)
    print(f"CSV file '{csv_file_name}' generated successfully with {len(tasks_data)} tasks.")
    print(f"Number of 'must-drop' tasks generated: {must_drop_tasks_generated}")


    print("\nFirst 20 tasks generated (showing progression examples and potential must-drops):")
    # Sort by Target_ID and then Request_Time to see progression clearly
    sorted_tasks = sorted(tasks_data, key=lambda x: (x[0], x[5]))
    for row in sorted_tasks[:20]:
        # Add a flag to easily identify must-drop tasks for printing
        is_must_drop_task = (row[4] == MUST_DROP_PRIORITY and
                             row[6] < row[5] + row[7] * 0.5) # Check for tight deadline
        print(f"Target ID: {row[0]}, Task ID: {row[1]}, Type: {row[2]}, Radar: {row[3]}, "
              f"Prio: {row[4]}, ReqTime: {row[5]:.2f}, Deadline: {row[6]:.2f}, Dur: {row[7]:.2f}, "
              f"InitP: {row[8]:.2f}, MaxP: {row[9]:.2f}, MaxD: {row[10]:.2f}, StageC: {row[11]} "
              f"{'(MUST-DROP)' if is_must_drop_task else ''}")

except IOError as e:
    print(f"Error writing CSV file: {e}")