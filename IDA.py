import gymnasium as gym
import time
import matplotlib.pyplot as plt

def manhattan_distance(state):
    row, col = divmod(state, 4)
    goal_row, goal_col = 3, 3
    return abs(goal_row - row) + abs(goal_col - col)

def ida_star(env, start_state, goal_state, heuristic, tau=600):
    start_time = time.time()

    def search(path, g, bound):
        if time.time() - start_time > tau:
            raise TimeoutError("Exceeded time threshold")
        node = path[-1]
        f = g + heuristic(node)
        if f > bound:
            return f
        if node == goal_state:
            return "FOUND"
        min_cost = float('inf')
        for action in range(env.action_space.n):
            env.unwrapped.s = node
            next_state, reward, done, truncated, _ = env.step(action)
            path.append(next_state)
            result = search(path, g + 1, bound)
            if result == "FOUND":
                return "FOUND"
            if isinstance(result, (int, float)) and result < min_cost:
                min_cost = result
            path.pop()
        return min_cost

    bound = heuristic(start_state)
    path = [start_state]
    while True:
        result = search(path, 0, bound)
        if result == "FOUND":
            return path
        if result == float('inf'):
            return None
        bound = result

def run_ida_star(num_runs=5, tau=600):
    times = []
    paths = []
    for i in range(num_runs):
        env = gym.make("FrozenLake-v1", is_slippery=False)
        print(f"\nRun {i+1}...")
        start_state, _ = env.reset()
        goal_state = env.observation_space.n - 1
        try:
            start_t = time.time()
            path = ida_star(env, start_state, goal_state, manhattan_distance, tau=tau)
            run_time = time.time() - start_t
            times.append(run_time)
            paths.append(path)
            if path is not None:
                print(f"Time: {run_time:.2f} s | Path length: {len(path)-1} | Path: {path}")
            else:
                print(f"Time: {run_time:.2f} s | No path found (search exhausted or bound too low).")
        except TimeoutError as e:
            print(str(e))
        finally:
            env.close()

    return times, paths

if __name__ == "__main__":
    import numpy as np

    num_runs = 5
    tau = 600  # 10 minutes

    times, paths = run_ida_star(num_runs, tau)
    avg_time = np.mean(times) if times else 0

    print("\nAverage Time:", avg_time)

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, num_runs + 1), times, color='skyblue')
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f"Avg Time: {avg_time:.2f}s")
    plt.xlabel("Run")
    plt.ylabel("Time (s)")
    plt.title("IDA* on FrozenLake: Time per Run")
    plt.legend()
    plt.show()
