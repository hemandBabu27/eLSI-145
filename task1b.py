import json
import time
from connector_task1b import CoppeliaClient

# ================= PID PARAMETERS =================

Kp = 1.2
Ki = 0.0
Kd = 0.5

MAX_SPEED = 4.0
MIN_SPEED = 0.0
BASE_SPEED = 2.0

# ================= Q-LEARNING PARAMETERS =================

ALPHA = 0.1
GAMMA = 0.9

Q_TABLE_FILE = "qtable_from_pid.json"
SAVE_EVERY = 5000

DEBUG = False

# ================= ACTIONS =================

ACTIONS = [
    "HARD_LEFT",
    "LEFT",
    "SLIGHT_LEFT",
    "STRAIGHT",
    "SLIGHT_RIGHT",
    "RIGHT",
    "HARD_RIGHT",
    "SEARCH_LEFT",
    "SEARCH_RIGHT",
    "CORNER_LEFT",
    "CORNER_RIGHT",
]

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

N_ACTIONS = len(ACTIONS)
N_STATES = 32

# Fixed-size Q-table (32 states × 11 actions)
q_table = [[0.0] * N_ACTIONS for _ in range(N_STATES)]

# ================= CONTROLLER VARIABLES =================

prev_error = 0.0
integral = 0.0
last_direction = 1

track_mode = None
candidate_mode = None
candidate_count = 0
CONFIRM_FRAMES = 5

# ================= RL VARIABLES =================

prev_state = None
prev_action = None
step_count = 0

# ===================================================

def get_state(track):
    return (
        (int(track[0]) << 4) |
        (int(track[1]) << 3) |
        (int(track[2]) << 2) |
        (int(track[3]) << 1) |
        int(track[4])
    )


def clamp(speed):
    if speed < MIN_SPEED:
        return MIN_SPEED
    if speed > MAX_SPEED:
        return MAX_SPEED
    return speed


def classify_action(left_speed, right_speed):

    diff = right_speed - left_speed

    if diff > 2.5:
        return ACTION_INDEX["HARD_RIGHT"]
    elif diff > 1.0:
        return ACTION_INDEX["RIGHT"]
    elif diff > 0.3:
        return ACTION_INDEX["SLIGHT_RIGHT"]
    elif diff < -2.5:
        return ACTION_INDEX["HARD_LEFT"]
    elif diff < -1.0:
        return ACTION_INDEX["LEFT"]
    elif diff < -0.3:
        return ACTION_INDEX["SLIGHT_LEFT"]

    return ACTION_INDEX["STRAIGHT"]


def compute_reward(position, is_corner, is_search):

    if is_search:
        return -5.0

    if is_corner:
        return 1.0

    return -abs(position)


def save_q_table():

    data = {
        "actions": ACTIONS,
        "q_table": q_table
    }

    with open(Q_TABLE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def control_loop(sensors):

    global prev_error, integral, last_direction
    global track_mode, candidate_mode, candidate_count
    global prev_state, prev_action, step_count

    lc = sensors["left_corner"]
    l = sensors["left"]
    m = sensors["middle"]
    r = sensors["right"]
    rc = sensors["right_corner"]

    sensor_values = [lc, l, m, r, rc]

    white_count = lc + l + m + r + rc
    black_count = 5 - white_count

    if track_mode is None:
        track_mode = "BLACK" if white_count > black_count else "WHITE"

    proposed = "BLACK" if white_count > black_count else "WHITE"

    if proposed != track_mode:

        if proposed == candidate_mode:
            candidate_count += 1
        else:
            candidate_mode = proposed
            candidate_count = 1

        if candidate_count >= CONFIRM_FRAMES:
            track_mode = candidate_mode
            candidate_count = 0

    else:
        candidate_mode = None
        candidate_count = 0

    if track_mode == "BLACK":
        track = [1 - s for s in sensor_values]
    else:
        track = sensor_values

    state = get_state(track)
    is_corner = False
    is_search = False
    position = 0.0

    t0, t1, t2, t3, t4 = track

    # ---------- Corner Detection ----------
    if t0 and t1 and not t3 and not t4:

        last_direction = -1
        left_speed = 0.0
        right_speed = 3.5

        action = ACTION_INDEX["CORNER_LEFT"]
        is_corner = True

    elif t3 and t4 and not t0 and not t1:

        last_direction = 1
        left_speed = 3.5
        right_speed = 0.0

        action = ACTION_INDEX["CORNER_RIGHT"]
        is_corner = True

    else:

        numerator = -2*t0 - t1 + t3 + 2*t4
        denominator = t0 + t1 + t2 + t3 + t4

        if denominator == 0:

            is_search = True

            if last_direction < 0:
                left_speed = 0.2
                right_speed = 3.0
                action = ACTION_INDEX["SEARCH_LEFT"]
            else:
                left_speed = 3.0
                right_speed = 0.2
                action = ACTION_INDEX["SEARCH_RIGHT"]

        else:

            position = numerator / denominator

            if position > 0:
                last_direction = 1
            elif position < 0:
                last_direction = -1

            error = -position
            integral += error
            derivative = error - prev_error

            correction = Kp * error + Ki * integral + Kd * derivative

            prev_error = error

            base_speed = max(1.0, BASE_SPEED - abs(error) * 0.5)

            left_speed = clamp(base_speed - correction)
            right_speed = clamp(base_speed + correction)

            action = classify_action(left_speed, right_speed)

    reward = compute_reward(position, is_corner, is_search)

    if prev_state is not None:

        q_prev = q_table[prev_state]
        q_next = q_table[state]

        target = reward + GAMMA * max(q_next)

        q_prev[prev_action] += ALPHA * (target - q_prev[prev_action])

    prev_state = state
    prev_action = action

    step_count += 1

    if step_count % SAVE_EVERY == 0:
        save_q_table()

    return left_speed, right_speed


def main():

    client = CoppeliaClient()

    print("Connecting...")
    client.connect()
    print("Connected")

    try:

        while True:

            sensors = client.receive_sensor_data()

            if sensors is not None:

                left_speed, right_speed = control_loop(sensors)
                client.send_motor_command(left_speed, right_speed)

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:

        save_q_table()
        client.close()

        print(
            f"Saved Q-table with {len(q_table)} states to {Q_TABLE_FILE}"
        )


if __name__ == "__main__":
    main()