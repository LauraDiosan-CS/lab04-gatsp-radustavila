
from start.heap import MyHeap
import random

from sys import maxsize

V = 4


# implementation of traveling Salesman Problem
def travellingSalesmanProblem(graph, s):
    # store all vertex apart from source vertex
    vertex = []
    for i in range(V):
        if i != s:
            vertex.append(i)

            # store minimum weight Hamiltonian Cycle
    min_path = maxsize

    while True:

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for i in range(len(vertex)):
            current_pathweight += graph[k][vertex[i]]
            k = vertex[i]
        current_pathweight += graph[k][s]

        # update minimum
        min_path = min(min_path, current_pathweight)

        if not next_permutation(vertex):
            break

    return min_path


# next_permutation implementation
def next_permutation(L):
    n = len(L)

    i = n - 2
    while i >= 0 and L[i] >= L[i + 1]:
        i -= 1

    if i == -1:
        return False

    j = i + 1
    while j < n and L[j] > L[i]:
        j += 1
    j -= 1

    L[i], L[j] = L[j], L[i]

    left = i + 1
    right = n - 1

    while left < right:
        L[left], L[right] = L[right], L[left]
        left += 1
        right -= 1

    return True


def find_the_path(graph, n, source, destination):
    heap = MyHeap()

    for i in range(n):
        if i != source:
            heap.push((graph[source][i], [source, i]))

    while True:
        cost, path = heap.pop()

        if destination == path[-1] or source == path[-1]:
            #path = map(lambda x : int(x + 1), path)
            return cost, path

        if len(path) == n:
            copy = path.copy()
            copy.append(source)
            heap.push((cost + graph[path[-1]][source], copy))

        for i in range(n):
            if i not in path:
                copy = path.copy()
                copy.append(i)
                heap.push((cost + graph[path[-1]][i], copy))



class Service(object):


    def __init__(self, repo):
        self._repo = repo


    def lab2(self):
        graph = self._repo.get_graph()
        n = self._repo.get_length()

        return find_the_path(graph, n, 0, 0)


    def lab4(self):

        POPULATION_SIZE = self._repo.get_length()
        graph = self._repo.get_graph()
        SOURCE = 0
        NUMB_OF_ELITE_CHROMOSOMES = 1
        TOURNAMENT_SELECTION_SIZE = 3
        MUTATION_RATE = 0.1
        cost1, TARGET_CHROMOSOME = find_the_path(graph, POPULATION_SIZE, SOURCE, SOURCE)
        del TARGET_CHROMOSOME[-1]

        print(cost1, TARGET_CHROMOSOME)


        class Chromosome:

            def __init__(self):
                self._genes=[]
                self._fitness=0
                self._genes.append(SOURCE)
                i = 1
                while i < len(TARGET_CHROMOSOME):

                    x = random.randint(0, POPULATION_SIZE - 1)
                    if x not in self._genes:
                        self._genes.append(x)
                    else:
                        while x in self._genes:
                            x = random.randint(0, POPULATION_SIZE - 1)
                        self._genes.append(x)
                    i += 1

            def get_genes(self):
                return self._genes

            def get_fitness(self):
                self._fitness = 0
                for i in range(POPULATION_SIZE):
                    if self._genes[i] == TARGET_CHROMOSOME[i]:
                        self._fitness += 1
                return self._fitness

            def _str_(self):
                return self._genes.__str__()




        class Population:
            def __init__(self, size):
                self._chromosomes = []
                i = 0
                while i < size:
                    self._chromosomes.append(Chromosome())
                    i += 1

            def get_chromosomes(self):
                return self._chromosomes

        class GeneticAlgorithm:

            @staticmethod
            def evolve(pop):
                return GeneticAlgorithm._mutate_population(GeneticAlgorithm._crossover_population(pop))

            @staticmethod
            def _crossover_population(pop):
                crossover_pop=Population(0)
                for i in range(NUMB_OF_ELITE_CHROMOSOMES):
                    crossover_pop.get_chromosomes().append(pop.get_chromosomes()[i])
                i = NUMB_OF_ELITE_CHROMOSOMES
                while i < POPULATION_SIZE:
                    chromosome1 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
                    chromosome2 = GeneticAlgorithm._select_tournament_population(pop).get_chromosomes()[0]
                    crossover_pop.get_chromosomes().append(GeneticAlgorithm._crossover_chromosomes(chromosome1, chromosome2))
                    i += 1

                return crossover_pop

            @staticmethod
            def _mutate_population(pop):
                for i in range (NUMB_OF_ELITE_CHROMOSOMES, POPULATION_SIZE):
                    GeneticAlgorithm._mutate_chromosome(pop.get_chromosomes()[i])
                return pop

            @staticmethod
            def _crossover_chromosomes(chromosome1, chromosome2):
                crossover_chrom = Chromosome()
                for i in range(1, len(TARGET_CHROMOSOME) - 1):
                    if random.random() >= 0.5:
                        crossover_chrom.get_genes()[i] = chromosome1.get_genes()[i]
                    else:
                        crossover_chrom.get_genes()[i] = chromosome2.get_genes()[i]

                return crossover_chrom

            @staticmethod
            def _mutate_chromosome(chromosome):
                for i in range(1, len(TARGET_CHROMOSOME) - 1):
                    if random.random() < MUTATION_RATE:
                        x = random.randint(1, POPULATION_SIZE - 1)
                        y = chromosome.get_genes()[i]
                        for k in range (1, len(TARGET_CHROMOSOME) - 1):
                            if chromosome.get_genes()[k] == x:
                                chromosome.get_genes()[k] = y
                        chromosome.get_genes()[i] = x


            @staticmethod
            def _select_tournament_population(pop):
                tournament_pop = Population(0)
                i = 0
                while i < TOURNAMENT_SELECTION_SIZE:
                    tournament_pop.get_chromosomes().append(pop.get_chromosomes()[random.randrange(0, POPULATION_SIZE)])
                    i += 1

                tournament_pop.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
                return tournament_pop




        def _print_population(pop, gen_number):
            print ("\n--------------------------------------")
            print ("Generatia #", gen_number, "| Fittest chromosome fitness:", pop.get_chromosomes()[0].get_fitness())
            print ("----------------------------------------")
            i=0
            for x in pop.get_chromosomes():
                print ("Chromosome  #", i, ":", x.get_genes(), "| Fitness: ", x.get_fitness())
                i += 1



        population = Population(POPULATION_SIZE)
        population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
        _print_population(population, 0)

        generation_number = 1

        while population.get_chromosomes()[0].get_fitness() < len(TARGET_CHROMOSOME):
            population = GeneticAlgorithm.evolve(population)
            population.get_chromosomes().sort(key=lambda x: x.get_fitness(), reverse=True)
            _print_population(population, generation_number)
            generation_number += 1


        print("\nTarget Chromosome: " + str(TARGET_CHROMOSOME) +"\nTarget fitness: " + str(POPULATION_SIZE) + "\n")
