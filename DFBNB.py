import sys
sys.setrecursionlimit(10000)

import gymnasium as gym
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_frames_as_gif(frames, path='./', filename='dfbnb_exploration.gif'):
    if not frames:
        print("No frames to save.")
        return
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    print("Saving the Animation....")
    print(len(frames))
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=80)
    anim.save(path + filename, writer='pillow', fps=30)
    print("Saved....")
    plt.close()

def manhattan_distance(state, goal_state, grid_size):
    x1, y1 = divmod(state, grid_size)
    x2, y2 = divmod(goal_state, grid_size)
    return abs(x1 - x2) + abs(y1 - y2)

def is_goal(state, env):
    grid_size = env.unwrapped.desc.shape[0]
    goal_state = grid_size * grid_size - 1
    return state == goal_state

def DFBNB(state, path, current_cost, best_cost, best_path, env, start_time, tau, frames, visited):
    if time.time() - start_time > tau:
        raise TimeoutError("Exceeded time threshold")
    if state in visited:
        return
    visited.add(state)
    
    frame = env.render()
    frames.append(frame)
    
    if is_goal(state, env):
        if current_cost < best_cost[0]:
            best_cost[0] = current_cost
            best_path[0] = path.copy()
            print(f"[INFO] Goal reached with cost {best_cost[0]}: {best_path[0]}")
        visited.remove(state)
        return
    
    grid_size = env.unwrapped.desc.shape[0]
    goal_state = grid_size * grid_size - 1
    
    for action in range(env.action_space.n):
        env.unwrapped.s = state
        next_state, reward, done, truncated, _ = env.step(action)
        if done:
            if reward == 1.0:
                if current_cost + 1 < best_cost[0]:
                    best_cost[0] = current_cost + 1
                    best_path[0] = path + [action]
                    print(f"[INFO] Goal reached with cost {best_cost[0]}: {best_path[0]}")
                continue
            else:
                continue
        h_value = manhattan_distance(next_state, goal_state, grid_size)
        if current_cost + 1 + h_value < best_cost[0]:
            DFBNB(next_state, path + [action], current_cost + 1, best_cost, best_path, env, start_time, tau, frames, visited)
    visited.remove(state)

def run_DFBNB(tau=600):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array")
    initial_state, _ = env.reset()
    best_cost = [float('inf')]
    best_path = [None]
    frames = []
    start_time = time.time()
    visited = set()
    
    try:
        DFBNB(initial_state, [], 0, best_cost, best_path, env, start_time, tau, frames, visited)
    except TimeoutError as e:
        print(e)
        run_time = tau
    else:
        run_time = time.time() - start_time
    
    env.close()
    return run_time, best_path[0], best_cost[0], frames

def main():
    num_runs = 5
    times = []
    best_costs = []
    best_paths = []
    
    for i in range(num_runs):
        print(f"\nRun {i+1}...")
        run_time, path, cost, frames_run = run_DFBNB(tau=600)
        times.append(run_time)
        best_costs.append(cost if cost != float('inf') else None)
        best_paths.append(path)
        print(f"Run {i+1}: Time: {run_time:.2f} s, Cost: {cost}, Path: {path}")
        
        if i == 0 and frames_run:
            save_frames_as_gif(frames_run, filename="dfbnb_exploration_run1.gif")
    
    successful_runs = [t for t in times if t < 600]
    avg_time = sum(successful_runs) / len(successful_runs) if successful_runs else 600
    print("\nAverage Time Taken:", avg_time)
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, num_runs + 1), times, color='skyblue', label="Run Time (s)")
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f"Average Time: {avg_time:.2f} s")
    plt.xlabel("Run Number")
    plt.ylabel("Time (s)")
    plt.title("Time Taken to Reach the Goal State per Run (DFBNB)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
