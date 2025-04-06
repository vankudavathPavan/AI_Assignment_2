import math
import random
import time
import matplotlib.pyplot as plt
import imageio

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

def hill_climbing_tsp(cities, dist_matrix, max_time=600, output_gif=False):
    start_time = time.time()
    n = len(cities)
    current_route = initial_route(n)
    current_distance = total_distance(current_route, dist_matrix)
    best_route = current_route.copy()
    best_distance = current_distance
    iterations = 0
    frames = []

    while time.time() - start_time < max_time:
        improved = False
        i = random.randint(0, n-2)
        j = random.randint(i+1, n-1)
        neighbor = current_route.copy()
        neighbor[i:j+1] = reversed(neighbor[i:j+1])
        neighbor_distance = total_distance(neighbor, dist_matrix)

        if neighbor_distance < current_distance:
            current_route = neighbor
            current_distance = neighbor_distance
            improved = True
            if neighbor_distance < best_distance:
                best_route = neighbor.copy()
                best_distance = neighbor_distance

        iterations += 1

        if output_gif and iterations % 100 == 0:
            plot_route(cities, current_route, iterations)
            plt.savefig(f'temp_{iterations}.png')
            plt.close()
            frames.append(imageio.imread(f'temp_{iterations}.png'))

        if not improved:
            break

    if output_gif and frames:
        imageio.mimsave('tsp_hill_climbing.gif', frames, duration=0.5)

    return best_route, best_distance, (time.time() - start_time)

def plot_route(cities, route, iteration):
    plt.figure()
    x = [cities[i][0] for i in route] + [cities[route[0]][0]]
    y = [cities[i][1] for i in route] + [cities[route[0]][1]]
    plt.plot(x, y, 'bo-')
    plt.title(f'Iteration: {iteration}')
    plt.xlabel('X')
    plt.ylabel('Y')

# Main execution
if __name__ == "__main__":
    cities = read_coordinates('coordinates.txt')
    dist_matrix = compute_distance_matrix(cities)
    n_runs = 20
    results = []

    for run in range(n_runs):
        print(f'Run {run + 1}')
        best_route, best_dist, time_taken = hill_climbing_tsp(
            cities, dist_matrix, output_gif=(run == 0))  # Generate GIF for first run
        results.append((best_dist, time_taken))
        print(f'Best Distance: {best_dist}, Time: {time_taken * 1000:.2f}ms')

    avg_distance = sum(r[0] for r in results) / n_runs
    avg_time = sum(r[1] for r in results) / n_runs
    print(f'\nAverage Distance: {avg_distance:.2f}')
    print(f'Average Time: {avg_time * 1000:.2f}ms')

    # Plotting the time taken for each run
    run_numbers = list(range(1, n_runs + 1))
    times = [r[1] * 1000 for r in results]  # Convert to milliseconds

    plt.figure()
    plt.plot(run_numbers, times, 'o-', color='green', label='Time per Run')
    plt.axhline(y=avg_time * 1000, color='red', linestyle='--', label=f'Average Time = {avg_time * 1000:.2f}ms')
    plt.title('Time Taken to Reach Optimum Across Runs')
    plt.xlabel('Run Number')
    plt.ylabel('Time (milliseconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('average_time_plot.png')
    plt.show()
