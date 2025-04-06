import math
import random
import time
import matplotlib.pyplot as plt
import imageio
import os

def read_coordinates(filename):
    cities = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            x, y = float(parts[1]), float(parts[2])
            cities.append((x, y))
    return cities

def compute_distance_matrix(cities):
    n = len(cities)
    dist_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                dist_matrix[i][j] = math.sqrt(dx**2 + dy**2)
    return dist_matrix

def initial_route(n):
    route = list(range(n))
    random.shuffle(route)
    return route

def total_distance(route, dist_matrix):
    total = 0
    for i in range(len(route)):
        j = (i + 1) % len(route)
        total += dist_matrix[route[i]][route[j]]
    return total

def simulated_annealing_tsp(cities, dist_matrix, max_time=60, output_gif=False):
    start_time = time.time()
    n = len(cities)
    current_route = initial_route(n)
    current_distance = total_distance(current_route, dist_matrix)
    best_route = current_route.copy()
    best_distance = current_distance
    temperature = 1000.0
    cooling_rate = 0.995
    iterations = 0
    frames = []

    while time.time() - start_time < max_time and temperature > 1e-8:
        i = random.randint(0, n - 2)
        j = random.randint(i + 1, n - 1)
        neighbor = current_route.copy()
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        neighbor_distance = total_distance(neighbor, dist_matrix)

        delta = neighbor_distance - current_distance
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_route = neighbor
            current_distance = neighbor_distance
            if neighbor_distance < best_distance:
                best_route = neighbor.copy()
                best_distance = neighbor_distance

        temperature *= cooling_rate
        iterations += 1

        if output_gif and iterations % 100 == 0:
            plot_route(cities, current_route, f'Iteration: {iterations}')
            filename = f'temp_{iterations}.png'
            plt.savefig(filename)
            plt.close()
            frames.append(imageio.imread(filename))

    if output_gif and frames:
        imageio.mimsave('tsp_simulated_annealing.gif', frames, duration=0.5)
        # Clean up temporary PNGs
        for file in os.listdir():
            if file.startswith("temp_") and file.endswith(".png"):
                os.remove(file)

    return best_route, best_distance, time.time() - start_time

def plot_route(cities, route, title):
    plt.figure(figsize=(8, 6))
    x = [cities[i][0] for i in route] + [cities[route[0]][0]]
    y = [cities[i][1] for i in route] + [cities[route[0]][1]]
    plt.plot(x, y, 'bo-', markersize=4)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)

def plot_time_graph(times, avg_time):
    runs = list(range(1, len(times) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(runs, times, 'o-', label='Time per Run (ms)', color='green')
    plt.axhline(y=avg_time, color='red', linestyle='--', label=f'Avg Time = {avg_time:.2f} ms')
    plt.title('Time Taken Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Time (milliseconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_per_run_plot.png')
    print("✅ Saved: time_per_run_plot.png")
    plt.show()

# Main
if __name__ == "__main__":
    cities = read_coordinates('coordinates.txt')
    dist_matrix = compute_distance_matrix(cities)

    n_runs = 20
    results = []

    for run in range(n_runs):
        print(f"Run {run + 1}")
        best_route, best_distance, time_taken = simulated_annealing_tsp(
            cities, dist_matrix,
            max_time=60,
            output_gif=(run == 0)  # Only first run makes GIF
        )
        results.append((best_distance, time_taken * 1000))  # Store time in ms
        print(f"Best Distance: {best_distance:.2f}, Time: {time_taken * 1000:.2f} ms")

    # Average stats
    avg_time = sum(t for _, t in results) / n_runs
    avg_distance = sum(d for d, _ in results) / n_runs
    print(f"\nAverage Distance: {avg_distance:.2f}")
    print(f"Average Time: {avg_time:.2f} ms")

    # Final best route plot
    plot_route(cities, best_route, f'Best Route (Distance: {best_distance:.2f})')
    plt.savefig('best_route_plot.png')
    print("✅ Saved: best_route_plot.png")
    plt.show()

    # Time vs Run graph
    plot_time_graph([t for _, t in results], avg_time)
