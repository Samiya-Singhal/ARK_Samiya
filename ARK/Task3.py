import cv2
import numpy as np
import networkx as nx
import random
import pygame
from scipy.spatial import KDTree
from queue import PriorityQueue
#This function creates the image into a grayscale image and with the help of a threshold value assigns all the pixels
#either black colour or white colour 
def load_maze(image_path):                                                   
    maze = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                        
    _, binary_maze = cv2.threshold(maze, 127, 255, cv2.THRESH_BINARY)
    return binary_maze
#This function gives us a specified no. of points or pixels in the image which are white
def sample_free_points(maze, num_samples):
    h, w = maze.shape
    free_points = []
    while len(free_points) < num_samples:
        x, y = random.randint(0, w-1), random.randint(0, h-1)
        if maze[y, x] == 255: 
            free_points.append((x, y))
    return free_points

def is_collision_free(p1, p2, maze):
    x1, y1 = p1
    x2, y2 = p2
    line = np.linspace([x1, y1], [x2, y2], num=100, dtype=int)
    return all(0 <= x < 471 and 0 <= y < 358 and maze[y, x] == 255 for x, y in line)

def build_roadmap(samples, maze, k=10):
    graph = nx.Graph()
    tree = KDTree(samples)
    for i, point in enumerate(samples):
        neighbors = tree.query(point, k+1)[1][1:]
        for neighbor_idx in neighbors:
            neighbor = samples[neighbor_idx]
            if is_collision_free(point, neighbor, maze):
                graph.add_edge(point, neighbor, weight=np.linalg.norm(np.array(point) - np.array(neighbor)))
    return graph

def find_shortest_path(graph, start, goal):
    try:
        return nx.shortest_path(graph, source=start, target=goal, weight='weight')
    except nx.NetworkXNoPath:
        return None

def visualize(maze, graph, path):
    color_maze = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)
    for edge in graph.edges:
        cv2.line(color_maze, edge[0], edge[1], (200, 200, 200), 1)
    if path:
        for i in range(len(path) - 1):
            cv2.line(color_maze, path[i], path[i+1], (0, 0, 255), 2)
    cv2.imshow('PRM Path', color_maze)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_simulation(path, maze):
    pygame.init()
    h, w = maze.shape
    print("Initializing Pygame window...")
    screen = pygame.display.set_mode((w, h))
    maze_surface = pygame.surfarray.make_surface(np.stack([maze] * 3, axis=-1))

    for point in path:
        screen.blit(maze_surface, (0, 0))  
        pygame.draw.circle(screen, (255, 0, 0), point, 5)  
        pygame.display.flip()
        pygame.time.delay(100)
    
    pygame.quit()

    


if __name__ == "__main__":
    maze_path = "maze.png"
    maze = load_maze(maze_path)
    num_samples = 5000
    start = (50, 300)
    goal = (100, 300)
    samples = [start, goal] + sample_free_points(maze, num_samples)
    graph = build_roadmap(samples, maze)
    path = find_shortest_path(graph, start, goal)
    
    if path:
        visualize(maze, graph, path)
        run_simulation(path, maze)
    else:
        print("No path found!")
