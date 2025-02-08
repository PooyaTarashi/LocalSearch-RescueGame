import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400
TILE_SIZE = 40
ROWS, COLS = HEIGHT // TILE_SIZE, WIDTH // TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rescue the Hostage - Local Search")

# Colors
WHITE = (240, 248, 255)
RED = (255, 69, 0)      # Hostage color
BLUE = (30, 144, 255)   # Player color
LIGHT_GREY = (211, 211, 211) # Background grid color
FLASH_COLOR = (50, 205, 50) # Victory flash color
BUTTON_COLOR = (50, 205, 50) # Button color
BUTTON_TEXT_COLOR = (255, 255, 255) # Button text color

# Load images for player, hostage, and walls
player_image = pygame.image.load("AI1.png")  
hostage_image = pygame.image.load("AI2.png")  
wall_images = [
    pygame.image.load("AI3.png"),
    pygame.image.load("AI4.png"),
    pygame.image.load("AI5.png")
]

# Resize images to fit the grid
wall_images = [pygame.transform.scale(img, (TILE_SIZE, TILE_SIZE)) for img in wall_images]
player_image = pygame.transform.scale(player_image, (TILE_SIZE, TILE_SIZE))
hostage_image = pygame.transform.scale(hostage_image, (TILE_SIZE, TILE_SIZE))

# Constants for recent positions
MAX_RECENT_POSITIONS = 10
GENERATION_LIMIT = 50
MUTATION_RATE = 0.1

# Function to generate obstacles
def generate_obstacles(num_obstacles):
    obstacles = []
    while len(obstacles) < num_obstacles:
        new_obstacle = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        if new_obstacle not in obstacles:  # Make sure obstacles are not overlapping
            obstacles.append(new_obstacle)
    obstacle_images = [random.choice(wall_images) for _ in obstacles]
    return obstacles, obstacle_images

# Function to start a new game
def start_new_game():
    global player_pos, hostage_pos, recent_positions, obstacles, obstacle_images, ga_ls, ga_counter
    obstacles, obstacle_images = generate_obstacles(20)
    recent_positions = []
    ga_ls = []
    ga_counter = 0

    # Generate player and hostage positions with a larger distance
    while True:
        player_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        hostage_pos = [random.randint(0, COLS-1), random.randint(0, ROWS-1)]
        distance = math.dist(player_pos, hostage_pos)
        if distance > 8 and player_pos not in obstacles and hostage_pos not in obstacles:
            break

# Gets the current position of the player and returns the list of its neighbors
def get_neighbors(player_pos, obstacles:list):
    x, y = player_pos
    width, height = COLS, ROWS
    neighbors = []
    for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and [nx, ny] not in obstacles:
            neighbors.append([nx, ny])

    return neighbors

def choose_successor(neighbors, hostage):
    mini = float('inf')
    chosen_successor = []
    for loc in neighbors:
        dist = abs(loc[0] - hostage[0]) + abs(loc[1] - hostage[1])
        if dist < mini:
            mini = dist
            chosen_successor = loc

    return chosen_successor


# Function to move the player closer to the hostage using Hill Climbing algorithm
def hill_climbing(player:list, hostage:list, obstacles) -> list:
    curr_value = abs(player[0] - hostage[0]) + abs(player[1] - hostage[1])
    neighbors = get_neighbors(player_pos=player, obstacles=obstacles)

    # print(neighbors)
    # successor = min(neighbors, key=lambda loc: abs(loc[0] - hostage[0]) + abs(loc[1] - hostage[1]))
    successor = choose_successor(neighbors, hostage)
    # print(successor, hostage, sep=' ')
    successor_value = abs(successor[0] - hostage[0]) + abs(successor[1] - hostage[1])
    if successor_value < curr_value:
        return successor
    return player


temperature = 100  # Initial temperature
cooling_rate = 0.99
min_temp = 1
# Function for Simulated Annealing
def simulated_annealing(player, hostage, obstacles):
    global temperature, cooling_rate, min_temp

    # Acceptance probability function
    def acceptance_probability(old_cost, new_cost, temp):
        if new_cost < old_cost:
            return 1.0
        return math.exp((old_cost - new_cost) / temp)

    current_position = player
    current_cost = abs(current_position[0] - hostage[0]) + abs(current_position[1] - hostage[1])

    if temperature > min_temp:
        neighbors = get_neighbors(current_position, obstacles)

        next_position = random.choice(neighbors)
        next_cost = abs(next_position[0] - hostage[0]) + abs(next_position[1] - hostage[1])

        if acceptance_probability(current_cost, next_cost, temperature) > random.random():
            current_position = next_position
            current_cost = next_cost

        temperature *= cooling_rate

    return current_position

# Function for Genetic Algorithm
def genetic_algorithm(player, hostage, obstacles):
    def choose_parents(genes):
        fitness_weights = list(map(fitness, genes))

        total_w = sum(fitness_weights)
        prob_distrib = [weight / total_w for weight in fitness_weights]

        selected_elements = random.choices(genes, weights=prob_distrib, k=20)

        return selected_elements

    # Fitness function
    def fitness(individual):
        return 1 / (len(individual))

    # Generate random population
    def generate_population():
        # Generates 20 solutions via DFS algorithm
        stack = [(player, [player])]
        population = []

        while stack and len(population) < population_size:
            current_position, path = stack.pop()

            if current_position == hostage:
                population.append(path)
                continue

            for successor in get_neighbors(current_position, obstacles):
                if successor not in path:
                    stack.append((successor, path + [successor]))

        return population

    # Crossover function
    def crossover(parent1, parent2):
        parent1_tuples = [tuple(point) for point in parent1]
        parent2_tuples = [tuple(point) for point in parent2]

        common_points = set(parent1_tuples) & set(parent2_tuples)

        if not common_points:
            for c1 in parent1:
                for c2 in parent2:
                    if c2 in get_neighbors(c1, obstacles):
                        idx1, idx2 = parent1.index(c1), parent2.index(c2)
                        new_path1 = parent1[:idx1 + 1] + parent2[idx2:]
                        new_path2 = parent2[:idx2 + 1] + parent1[idx1:]

                        return new_path1, new_path2

        pivot = random.choice(list(common_points))
        idx1, idx2 = parent1.index([pivot[0], pivot[1]]), parent2.index([pivot[0], pivot[1]])
        new_path1 = parent1[:idx1] + parent2[idx2:]
        new_path2 = parent2[:idx2] + parent1[idx1:]

        return new_path1, new_path2

    # Mutation function
    def mutate(individual):
        new_path = individual[:]

        for i in range(len(new_path)):
            for j in range(len(new_path)):
                if i >= j:
                    break

                if new_path[j] in get_neighbors(new_path[i]):
                    new_path = new_path[:i + 1] + new_path[j:]

        return new_path


    population_size = 20
    generations = 50
    curr_generation = generate_population()

    for _ in range(generations):
        # Select 40 genes from the curr_gen
        selected_parents = choose_parents(curr_generation)

        # Crossover two two
        new_genes = []
        i = 0
        while i < len(selected_parents) // 2:
            child1, child2 = crossover(selected_parents[i], selected_parents[i + 1])
            new_genes.extend([child1, child2])
            i += 2

        # mutate the each one
        for gene in new_genes:
            gene = mutate(gene)

        curr_generation = curr_generation

    # Return the best individual
    return max(curr_generation, key=fitness)


#Objective: Check if the player is stuck in a repeating loop.
def in_loop(recent_positions, player):
    return player in recent_positions

#Objective: Make a random safe move to escape loops or being stuck.
def random_move(player, obstacles):
    neighbors = get_neighbors(player_pos=player, obstacles=obstacles)
    return random.choice(neighbors)

#Objective: Update the list of recent positions. 
def store_recent_position(recent_positions:list, new_player_pos, max_positions=MAX_RECENT_POSITIONS):
    recent_positions.append(new_player_pos)
    if len(recent_positions) > max_positions:
        recent_positions.pop(0)

# Function to show victory flash
def victory_flash():
    for _ in range(5):
        screen.fill(FLASH_COLOR)
        pygame.display.flip()
        pygame.time.delay(100)
        screen.fill(WHITE)
        pygame.display.flip()
        pygame.time.delay(100)

# Function to show a button and wait for player's input
def show_button_and_wait(message, button_rect):
    font = pygame.font.Font(None, 36)
    text = font.render(message, True, BUTTON_TEXT_COLOR)
    button_rect.width = text.get_width() + 20
    button_rect.height = text.get_height() + 10
    button_rect.center = (WIDTH // 2, HEIGHT // 2)
    pygame.draw.rect(screen, BUTTON_COLOR, button_rect)
    screen.blit(text, (button_rect.x + (button_rect.width - text.get_width()) // 2,
                       button_rect.y + (button_rect.height - text.get_height()) // 2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False

# Function to get the algorithm choice from the player
def get_algorithm_choice():
    print("Choose an algorithm:")
    print("1: Hill Climbing")
    print("2: Simulated Annealing")
    print("3: Genetic Algorithm")

    while True:
        choice = input("Enter the number of the algorithm you want to use (1/2/3): ")
        if choice == "1":
            return hill_climbing
        elif choice == "2":
            return simulated_annealing
        elif choice == "3":
            return genetic_algorithm
        else:
            print("Invalid choice. Please choose 1, 2, or 3.")

# Main game loop
running = True
clock = pygame.time.Clock()
start_new_game()
button_rect = pygame.Rect(0, 0, 0, 0)

# Get the algorithm choice from the player
chosen_algorithm = get_algorithm_choice()

ga_counter = 0
ga_ls = []

steps = 0

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Perform the chosen algorithm step
    if chosen_algorithm == genetic_algorithm:
        if ga_counter == 0:
            ga_ls = chosen_algorithm(player_pos, hostage_pos, obstacles)
            print(ga_ls)
            print(len(ga_ls))
        new_player_pos = ga_ls[ga_counter]
        ga_counter += 1
    else:
        new_player_pos = chosen_algorithm(player_pos, hostage_pos, obstacles)

    # Check for stuck situations
    if chosen_algorithm != genetic_algorithm and (new_player_pos == player_pos or in_loop(recent_positions, new_player_pos)):
        # Perform a random move when stuck
        new_player_pos = random_move(player_pos, obstacles)


    # Update recent positions
    store_recent_position(recent_positions, new_player_pos)
    # Update player's position
    player_pos = new_player_pos

    # Draw the grid background
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, LIGHT_GREY, rect, 1)

    # Draw obstacles
    for idx, obs in enumerate(obstacles):
        obs_rect = pygame.Rect(obs[0] * TILE_SIZE, obs[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(obstacle_images[idx], obs_rect)

    # Draw player
    player_rect = pygame.Rect(player_pos[0] * TILE_SIZE, player_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    screen.blit(player_image, player_rect)

    # Draw hostage
    hostage_rect = pygame.Rect(hostage_pos[0] * TILE_SIZE, hostage_pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
    screen.blit(hostage_image, hostage_rect)

    # Check if player reached the hostage
    if player_pos == hostage_pos:
        print(steps)
        print("Hostage Rescued!")
        victory_flash()  # Show the victory flash
        show_button_and_wait("New Game", button_rect)
        start_new_game()

    # Update the display
    pygame.display.flip()
    clock.tick(5)  # Lower frame rate for smoother performance

pygame.quit()
