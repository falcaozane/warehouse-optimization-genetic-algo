import numpy as np
import matplotlib.pyplot as plt
import random
import io
import base64
import matplotlib.colors as mcolors

class Warehouse:
    def __init__(self, length, width, height):
        self.length = length
        self.width = width
        self.height = height
        self.space = np.zeros((length, width, height), dtype=int)
        self.entrances = [] 

    def add_entrance(self, location):
        self.entrances.append(location)
        x, y = location
        self.space[x, y ,:] = 1

    def block_area(self, initial, dimensions):
        x, y, z = initial
        blocked_length, blocked_width, blocked_height = dimensions
        self.space[ x:x + blocked_length, y:y + blocked_width , z:z + blocked_height] = 2

    def display_warehouse(self):
        fig, axes = plt.subplots(self.height, 1, figsize=(10, 5*self.height))
        cmap = plt.cm.get_cmap('viridis', 5)  # Use a colormap with 5 distinct colors

        for z in range(self.height):
            ax = axes[z]
            ax.set_title(f"Level {z + 1}")
            level_matrix = np.zeros((self.length, self.width))

            for x in range(self.length):
                for y in range(self.width):
                    if (x, y) in self.entrances:
                        level_matrix[x, y] = 1  # Entrance
                    elif self.space[x, y, z] == 2:
                        level_matrix[x, y] = 2  # Obstacle
                    elif self.space[x, y, z] == 3:
                        level_matrix[x, y] = 3  # Path
                    elif self.space[x, y, z] == 0:
                        level_matrix[x, y] = 0 # Empty space
                    else:
                        level_matrix[x,y] = self.space[x, y, z]

            im = ax.imshow(level_matrix, cmap=cmap, vmin=0, vmax=4, origin='upper')
            ax.set_xticks(np.arange(self.width))
            ax.set_yticks(np.arange(self.length))
            ax.set_xticklabels([chr(65 + x) for x in range(self.width)])  # Label columns with letters
            ax.set_yticklabels([str(y + 1) for y in range(self.length)])  # Label rows with numbers
            ax.grid(which="major", color="gray", linestyle='-', linewidth=2)

        # Add a legend to explain the color codes
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_yticklabels(['Entrance', 'Obstacle', 'Path', 'Empty Space'])

        # Convert the figure to an HTML string for Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plot_html = f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode()}">'
        plt.close(fig)

        return plot_html
        
    def add_products(self, products):
        for product, location in products.items():
            if isinstance(product, Product):
                x, y, z = location
                self.space[x, y, z] = product.id # Assuming str(product) gives product ID


    def make_path(self, start, end):
        x1, y1 = start
        x2, y2 = end
        while int(x1) != int(x2) and int(y1) != int(y2):
            self.space[ int(x1), int(y1),:] = 3
            x1 += int((x2-x1)//abs(x2-x1))
            y1 += int((y2-y1)//abs(y2-y1))
        while int(x1)!=x2:
            self.space[int(x1), int(y1) , :] = 3
            x1 += int((x2-x1)//abs(x2-x1))
        while int(y1)!=y2:
            self.space[int(x1), int(y1), :] = 3
            y1 += int((y2-y1)//abs(y2-y1))
        self.space[int(x2), int(y2), :] = 3

class Product:
    def __init__(self, height, length, width, fragile, demand,id , stock):
        self.height = height
        self.length = length
        self.width = width
        self.fragile = fragile
        self.demand = demand
        self.stock = stock
        self.id = id

def initialize_population(warehouse, products, population_size):
    population = []
    for _ in range(population_size):
        chromosome = {}
        all_products_placed = True
        for product in products:
            placed = False
            attempts = 0
            max_attempts = 1000  # Maximum number of attempts to find a valid position
            while not placed and attempts < max_attempts:
                attempts += 1
                # Randomly select a position within the warehouse space
                x = random.randint(0, warehouse.length - 1)
                y = random.randint(0, warehouse.width - 1)
                z = random.randint(0, warehouse.height - 1)
                # Check if the selected position is not blocked or part of the path
                if warehouse.space[x,y,z] == 0 :
                    chromosome[product] = (x, y, z)
                    placed = True
            if not placed:
                # Find any available position in the warehouse space
                for z in range(warehouse.height):
                    for x in range(warehouse.length):
                        for y in range(warehouse.width):
                            if warehouse.space[x, y,z] == 0:
                                chromosome[product] = (x, y, z)
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
            if not placed:
                all_products_placed = False
                print("Not Enough Space in the Warehouse")
                return False
        if all_products_placed:
            population.append(chromosome)
    return population

def evaluate_population_fitness(population, warehouse, products):
    for chromosome in population:
        fitness = 0
        fragility_penalty = 0
        demand_distance_score = 0
        occupied_spaces = np.sum(warehouse.space != 0)
        total_spaces = warehouse.length * warehouse.width * warehouse.height
        remaining_spaces = total_spaces - occupied_spaces

        # Iterate through each product placement
        for product, position in chromosome.items():
            if product == "fitness":
                break
            x, y, z = position

            # Check fragility constraint
            if product.fragile:
                # Check if there is another product stacked above
                for z_above in range(z + 1, warehouse.height):
                    if warehouse.space[x, y , z_above] != 0:
                        fragility_penalty += 10
                        break

            # Calculate distance from the entrance for demand forecasting
            entrance_distance = min([abs(x - ex) + abs(y - ey) for ex, ey in warehouse.entrances])

            # Calculate the demand_distance_score based on the relationship between demand, stock, path distance, and available space
            if remaining_spaces > (total_spaces * 0.2):  # If more than 20% of the space is available
                # Prefer placing all products near the entrance
                if product.demand > product.stock:
                    demand_distance_score += (product.demand - product.stock) / ((entrance_distance  + 1) ** 2)
                else:
                    demand_distance_score += (product.stock - product.demand) / ((warehouse.length + warehouse.width - entrance_distance + 1) ** 2)
            else:
                # If limited space, prioritize placing high demand and low stock products near the entrance 
                if product.demand > product.stock:
                    demand_distance_score += (product.demand - product.stock) / ((entrance_distance + 1) ** 2)
                else:
                    demand_distance_score += (product.stock - product.demand) / (warehouse.length + warehouse.width - entrance_distance+ 1)

        # Calculate the overall fitness score
        fitness = demand_distance_score - fragility_penalty

        # Update the fitness value for the individual
        chromosome['fitness'] = fitness                    

def select_individuals(population, tournament_size=5):
    # Sort the population by fitness in descending order
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)

    # Select two individuals using tournament selection
    selected_individuals = []
    for _ in range(2):
        # Randomly select tournament_size individuals from the population
        tournament = [sorted_population[random.randint(0, len(sorted_population) - 1)] for _ in range(tournament_size)]

        # Select the fittest individual from the tournament
        selected_individual = max(tournament, key=lambda x: x['fitness'])
        selected_individuals.append(selected_individual)

    return selected_individuals

def crossover(selected_individuals):
    # Unpack the selected individuals
    parent1, parent2 = selected_individuals

    # Create empty offspring chromosomes
    offspring1 = {}
    offspring2 = {}

    # Determine the crossover point randomly
    crossover_point = random.randint(1, len(parent1) - 2)

    # Perform crossover
    for i, (product, position) in enumerate(parent1.items()):
        if i < crossover_point:
            offspring1[product] = position
            offspring2[product] = parent2[product]
        else:
            offspring1[product] = parent2[product]
            offspring2[product] = position

    return [offspring1, offspring2]

def mutate(warehouse,offspring, mutation_rate=0.5):
    mutated_offspring = []
    for individual in offspring:
        mutated_individual = {}
        for product, position in individual.items():
            if product != "fitness":
                x, y, z = position

                # Mutate each gene (position) with the given mutation rate
                if random.random() < mutation_rate:
                    # Randomly select a new position within the warehouse space
                    new_x = random.randint(0, warehouse.length - 1)
                    new_y = random.randint(0, warehouse.width - 1)
                    new_z = random.randint(0, warehouse.height - 1)

                    # Check if the new position is not blocked or part of the path
                    while warehouse.space[new_x, new_y,new_z] ==0:
                        new_x = random.randint(0, warehouse.length - 1)
                        new_y = random.randint(0, warehouse.width - 1)
                        new_z = random.randint(0, warehouse.height - 1)

                    mutated_individual[product] = (new_x, new_y, new_z)
                else:
                    mutated_individual[product] = position
            else:
                # If the key is 'fitness', copy it to the mutated_individual
                mutated_individual[product] = individual[product]

        mutated_offspring.append(mutated_individual)
    return mutated_offspring

def replace_population(population, offspring, replacement_rate=0.8):
    # Sort the population by fitness in descending order
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)

    # Calculate the number of individuals to be replaced
    num_replacements = int(len(population) * replacement_rate)

    # Replace the least fit individuals in the population with the offspring
    new_population = sorted_population[:-num_replacements] + offspring[:num_replacements]

    return new_population

def extract_best_solution(population):
    # Sort the population by fitness in descending order
    sorted_population = sorted(population, key=lambda x: x['fitness'], reverse=True)

    # The first individual in the sorted population is the best solution
    best_solution = sorted_population[0]

    return best_solution



def space_allocation_genetic_algorithm(warehouse, products, population_size, generations):
    # Initialization
    population = initialize_population(warehouse, products, population_size)
    if not population:
        return False
    print("initial population is ", population)
    for gen in range(generations):
        # Fitness evaluation
        evaluate_population_fitness(population, warehouse, products)
        print(" The fittness allocated population is ", population)
        
        # Selection
        selected_individuals = select_individuals(population)
        # print("The selected products for crossover is ", selected_individuals)
        
        # Crossover
        offspring = crossover(selected_individuals)
        # print("The offsprings are ", offspring)
        
        # Mutation
        m_offspring = mutate(warehouse,offspring)
        # print("Mutated Offsprings are ", m_offspring)
        
        # Replacement
        population = replace_population(population, m_offspring)
        print("the New population is ", population)
    
    # Final solution extraction
    best_solution = extract_best_solution(population)
    return best_solution

