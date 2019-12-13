import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from sklearn.neighbors import NearestNeighbors
#%%
#Hokusin Tyou: class Device, AHPsystem, Population, Chromosome, Gene, Algorithm with their subclasses

#caution: AHPsystem.average_availability() is now empty
#        SPEA2.density_information() is now empty
#        The truncation in SPEA2 is incomplete


#notes should be read before reading the code
#problem description
#1 object: min cost, max average availability
#2 constraints: cost<=1.2e6, average availability>=0.9, horizon=40 years
#3 variables: x[0]~x[9] ,representing the maintenance intervals of 10 devices
#4 we will use 01 code to represent x
#5 the time unit will be hour, please transform other units like year to hour
horizon=40*365*24 #350400

#6 the largest length of one Gene(the maintenance interval) will be manually set here
length_of_gene=19 #2**18=262144

#7 the object values are saved in a nparray of shape(solutions,2). 2 represent cost.(-availability) for minimization

#8 the non-dominance level will begin from 1

#9 infinity=1e10
INFINITY=1e10

#10
population_size=100
external_population_size=50

#there still exists some problems to be considered

#1 what to do with infeasible solutions? I chose to ignore them during mid-process.
# Professor Li said that we set the rank of unfeasible solutions to infinity

#2 SPEA2: what to do with environmental_selection when len(selected)>Ne?

#3 it seems that the cost is limited to the initialization


#%%

class Device():
    def __init__(self,maintenance_cost):
        self.maintenance_cost=maintenance_cost

class Heater(Device):
    def __init__(self,failure_rate,maintenance_cost):
        super().__init__(maintenance_cost=maintenance_cost)
        self.failure_rate=failure_rate

class Heater_no6(Heater):
    def __init__(self):
        super().__init__(failure_rate=3.48e-7,maintenance_cost=3.60e4)

class Heater_no7(Heater):
    def __init__(self):
        super().__init__(failure_rate=3e-7,maintenance_cost=4.32e4)

class Valve(Device):
    def __init__(self,demand_failure_probability,maintenance_cost):
        super().__init__(maintenance_cost=maintenance_cost)
        self.demand_failure_probability=demand_failure_probability

class Valve01(Valve):
    def __init__(self):
        super().__init__(demand_failure_probability=2.02e-4,maintenance_cost=1.42e3)

class Valve03(Valve):
    def __init__(self):
        super().__init__(demand_failure_probability=5.03e-4,maintenance_cost=1.01e3)

class Valve_main_bypass(Valve):
    def __init__(self):
        super().__init__(demand_failure_probability=2.60e-4, maintenance_cost=1.02e3)

class Valve_standby_bypass(Valve):
    def __init__(self):
        super().__init__(demand_failure_probability=2.10e-4, maintenance_cost=2.80e3)


class AHPsystem:
    def __init__(self,horizon=horizon,lower_bound_of_availability=0.95,upper_bound_of_cost=1.2e6):
        self.horizon=horizon
        self.lower_bound_of_availability=lower_bound_of_availability
        self.upper_bound_of_cost=upper_bound_of_cost
        self.devices=list()
        self.devices.append(Heater_no6())
        self.devices.append(Heater_no7())
        self.devices.append(Heater_no6())
        self.devices.append(Heater_no7())
        self.devices.append(Valve01())
        self.devices.append(Valve03())
        self.devices.append(Valve01())
        self.devices.append(Valve03())
        self.devices.append(Valve_main_bypass())
        self.devices.append(Valve_standby_bypass())

    '''
    #use x[0]-x[9] to represent maintenance intervals
    def get_intervals(self,x):
        if len(x)!=10:
            print("error")
            return
        else:
            self.intervals=x
    '''
    def maintenance_cost(self,x):
        #self.get_intervals(x)
        #x=self.intervals
        if len(x)!=10:
            print("error")
            return
        cost=0
        for i in range(10):
            cost+=self.horizon/x[i]*self.devices[i].maintenance_cost
        return cost

    def average_availability(self,x):#??????
        #self.get_intervals(x)

        return 1

#%%
class Population:# the Population save the solutions in a list
    def __init__(self,population_size=100,empty=False):
        if empty==False:
            self.size=population_size
            self.chromosomes=list()
            for i in range(self.size):
                self.chromosomes.append(Chromosome())
            self.solutions=self.get_solutions()
        else:
            self.size=population_size
            self.chromosomes=list()

    def get_solutions(self):
        solutions=list()
        for i in range(self.size):
            solutions.append(self.chromosomes[i].convert_to_intervals())
        return solutions
    '''
    def crossover(self,crossover_rate=1):#create offspring Population of same size
        number_to_crossover=int(self.size*crossover_rate/2)*2# transform into an even number

        #the criteria for chosen?????
        seq=np.random.choice(np.arange(self.size),size=number_to_crossover,replace=False)
        for i in range(number_to_crossover):
            no=seq[i]
        for i in range(int(number_to_crossover)):
            no1=seq[2*i]
            no2=seq[2*i+1]
            self.chromosomes[no1].crossover(self.chromosomes[no2])
    '''


class Chromosome:
    def __init__(self,number_of_genes=10,code=None,length_of_gene=19):
        self.number_of_genes=number_of_genes
        if code==None:
            self.genes=list()
            self.code=""
            for i in range(self.number_of_genes):
                self.genes.append(Gene())
                self.code+=self.genes[i].code
            self.length=len(self.code)
            self.gene_length=self.genes[0].length
        else:
            if(len(code)!=number_of_genes*length_of_gene):
                print("error")
                return
            self.code=code
            self.length=len(self.code)
            self.genes=list()
            self.gene_length=length_of_gene
            self.update()

    def update(self):
        if len(self.genes)==0:
            for i in range(self.number_of_genes):
                self.genes.append(Gene(code=self.code[i*self.gene_length:(i+1)*self.gene_length]))
        else:
            for i in range(self.number_of_genes):
                self.genes[i].code=self.code[i*self.gene_length:(i+1)*self.gene_length]

    def mutation(self,mutation_rate=0.1,mutation_size=1):
        if (np.random.random()>mutation_rate):
            return
        mutation_points=random.sample(range(self.length), mutation_size)
        list_code=list(self.code)
        for mutation_point in mutation_points:
            list_code[mutation_point]=str(1-int(list_code[mutation_point]))
            self.code=''.join(list_code)
        self.update()

    def crossover(self,other):
        point=np.random.randint(self.length)
        a=self.code
        b=other.code
        new_code1=a[0:point]+b[point:]
        new_code2=b[0:point]+a[point:]
        return [new_code1,new_code2]
        '''
        self.update()
        other.update()
        '''

    def convert_to_intervals(self):
        self.update()
        intervals=list()
        for i in range(self.number_of_genes):
            intervals.append(self.genes[i].convert_to_interval())
        return intervals

    def get_cost(self,AHPsystem):
        intervals=self.convert_to_intervals()
        cost=AHPsystem.maintenance_cost(intervals)
        return cost

    def get_availability(self,AHPsystem):
        intervals=self.convert_to_intervals()
        availability=AHPsystem.average_availability(intervals)
        return availability

class Gene:
    def __init__(self,length=length_of_gene,code=None):
        self.length=length
        if code==None:
            self.code=""
            for i in range(length):
                self.code+=random.choice("01")
        else:
            if length==len(code):
                self.code=code
            else:
                print("error")
                return

    def convert_to_interval(self):
        interval=int(self.code,2)
        self.interval=interval
        return interval


#%%
class Algorithm:
    def __init__(self,object_number=2,population_size=100):
        self.object_number=object_number
        self.population_size=population_size
        self.initialize()

    def initialize(self):
        self.Population=Population(self.population_size)

    #this function is to return an array indicating whether the solutions are non-dominated
    def is_non_dominated(self,objects):#default minimization
        #objects is of shape(solutions,2) 2 for cost,(-availability)

        #the fantastic code below has referred to https://stackoverflow.com/a/40239615
        is_non_dominated=np.ones(objects.shape[0], dtype = bool)

        #hokusin: the loop below is to: first set all solutions nondominated=true, then for every nondominated solution
        #check if other now nondominated==true solutions is smaller than it in any object. if not, set the dominated one nondominated=false
        for i, c in enumerate(objects):
            if is_non_dominated[i]:
                is_non_dominated[is_non_dominated]=np.any(objects[is_non_dominated]<c, axis=1)  # Keep any point with a lower cost
                is_non_dominated[i]=True  # And keep self
        return is_non_dominated
        ###

        #this function aims to return an array indicating the rank of each solution
        #i am not sure if this function can work without bug

    def non_dominance_ranking(self, objects):
        objects_copy=copy.copy(objects)  #to prevent mysterious errors, we always use copy
        n=objects_copy.shape[0]  #n solutions
        #number_not_classified=np.arange(n) #record the unclassified objects
        rankings=np.zeros(n, dtype="int")
        #objects_copy=np.insert(objects_copy,objects_copy.shape[1], values=rankings, axis=1)
        i=1
        number_not_classified=np.arange(n)
        #remaining_objects=copy.copy(objects_copy)
        #remaining_objects_copy=copy.copy(objects_copy)
        while np.any(rankings==0):  #all solutions has not been assigned a ranking
            remaining_objects=objects_copy[number_not_classified]
            results=self.is_non_dominated(remaining_objects)

            remaining_rankings=rankings[number_not_classified]
            remaining_rankings[results]=i
            rankings[number_not_classified]=remaining_rankings

            #objects_copy[:,-1]=
            number_not_classified=np.where(rankings==0)  #this line has problems!!!

            i+=1
        return rankings

    def final_solutions(self,parent_population,system):
        objects=np.empty(shape=(parent_population.size, 2))
        for i in range(objects.shape[0]):
            objects[i, 0]=parent_population.chromosomes[i].get_cost(system)
            objects[i, 1]=(-1)*parent_population.chromosomes[i].get_availability(system)
        rankings=self.non_dominance_ranking(objects)
        solutions=np.array(parent_population.chromosomes)[rankings==1]
        return solutions


class NSGA2(Algorithm):
    def __init__(self,object_number=2,population_size=100):
        super().__init__(object_number,population_size)


    #this function's input is one array,and output is a distance array of the same length
    def crowding_distance_one_object(self,object):
        object_copy=copy.copy(object)
        distances=np.zeros(len(object_copy),dtype="float")
        temporary_distances=np.zeros(len(object_copy),dtype="float")
        seq=np.argsort(object_copy)
        sorted_object=np.sort(object_copy)

        maxobject=object_copy.max()
        minobject=object_copy.min()
        difference=maxobject-minobject

        temporary_distances[0]=INFINITY
        temporary_distances[-1]=INFINITY
        for i in range(1,len(seq)-1):
            temporary_distances[i]=(sorted_object[i+1]-sorted_object[i-1])/difference

        for i in range(len(seq)):
            distances[seq[i]]=temporary_distances[i]

        return distances
        '''
        origin=np.empty(len(seq))
        for i in range(len(seq)):
            origin[seq[i]]=aaa[i]
        '''

    #this function aims to return an array indicating the distance of each solution
    #i am not sure if this function can work without bug
    def crowding_distance(self,objects):
        objects_copy=objects
        number_of_objects=objects_copy.shape[-1]
        distances=np.zeros(len(objects_copy),dtype="float")
        for i in range(number_of_objects):
            distances+=self.crowding_distance_one_object(objects_copy[:,i])

        return distances

    #this function returns an array of selected parents
    def crowded_tournament_selection(self,rankings,distances,numbers_of_winners=None):
        number_of_candidates=len(rankings)
        if numbers_of_winners==None:
            length=len(rankings)
            if len(rankings)!=len(distances):
                print("error!")
                return
            numbers_of_winners=len(rankings)
        else:
            pass
            #numbers_of_winners=numbers_of_winners
        parents=list()
        while len(parents)<numbers_of_winners:
            candidate1,candidate2=np.random.choice(number_of_candidates,size=2,replace=False)
            if rankings[candidate1]>rankings[candidate2]:
                parents.append(candidate2)
            elif rankings[candidate2]>rankings[candidate1]:
                parents.append(candidate1)
            else:
                if distances[candidate1]>distances[candidate2]:
                    parents.append(candidate1)
                else:
                    parents.append(candidate2)
        parents=np.array(parents)
        return parents

    def enviromental_selection(self,rankings,distances,population_size):
        #selected=list()
        i=1
        index=np.lexsort(keys=[-distances,rankings])
        return index[0:population_size]

    def nsga2(self,t=20):  #main body, t= generation counter
        parent_population=Population(population_size=100)
        system=AHPsystem()
        average_cost=list()
        average_minus_availability=list()
        min_cost=list()
        min_minus_availability=list()

        for round in range(t):
            #calculate the object function value of all solutions
            #caution: we use (-availability)
            objects=np.empty(shape=(parent_population.size,2))
            for j in range(objects.shape[0]):
                objects[j,0]=parent_population.chromosomes[j].get_cost(system)
                objects[j,1]=(-1)*parent_population.chromosomes[j].get_availability(system)

            #track the cost and availability
            average_cost.append(np.mean(objects[:,0]))
            average_minus_availability.append(np.mean(objects[:,1]))
            min_cost.append(np.min(objects[:,0]))
            min_minus_availability.append(np.min(objects[:,1]))

            #get the rankings and distances based on object values
            rankings=self.non_dominance_ranking(objects)
            distances=self.crowding_distance(objects)

            #select out the parents for crossover
            parents=self.crowded_tournament_selection(rankings,distances)

            #perform crossover
            offspring_population=Population(empty=True)
            for i in range(0,len(parents),2):
                new_code1,new_code2=parent_population.chromosomes[parents[i]].crossover(
                    parent_population.chromosomes[parents[i+1]]
                )
                offspring_population.chromosomes.append(Chromosome(code=new_code1))
                offspring_population.chromosomes.append(Chromosome(code=new_code2))
            #mutation
            for i in range(offspring_population.size):
                offspring_population.chromosomes[i].mutation()

            #calculate offspring's object values
            objects_offspring=np.empty(shape=(offspring_population.size, 2))
            for j in range(objects_offspring.shape[0]):
                objects_offspring[j, 0]=offspring_population.chromosomes[j].get_cost(system)
                objects_offspring[j, 1]=(-1)*offspring_population.chromosomes[j].get_availability(system)

            #combine parents and offspring
            objects_all=np.vstack((objects,objects_offspring))
            #get the rankings and distances based on object values
            rankings_all=self.non_dominance_ranking(objects_all)
            distances_all=self.crowding_distance(objects_all)

            #select out new parents
            new_parents=self.enviromental_selection(rankings_all, distances_all,population_size=parent_population.size)
            from_parents=new_parents[new_parents<len(objects)]
            from_offspring=new_parents[new_parents>=len(objects)]-len(objects)
            if (len(from_parents)+len(from_offspring))!=parent_population.size:
                print("error")
                return
            new_parent_population=Population(population_size=population_size,empty=True)
            for i in range(len(from_parents)):
                new_parent_population.chromosomes.append(parent_population.chromosomes[from_parents[i]])
            for i in range(len(from_offspring)):
                new_parent_population.chromosomes.append(offspring_population.chromosomes[from_offspring[i]])
            parent_population=new_parent_population

        #output the history and final nondominated solutions


        solutions=self.final_solutions(parent_population,system)
        return [average_cost,average_minus_availability,min_cost,min_minus_availability,solutions]


class SPEA2(Algorithm):

    #caution: I made a mistake here. Though I used the name"parent_population", but in fact in SPEA2 they do not
    #play the role of parents. Actually it's external population who crossover and mutate.
    #here, parent_population simply refer to Pt in PPT, while external_population = Et

    def __init__(self, object_number=2, population_size=100,external_population_size=50):
        super().__init__(object_number, population_size)
        self.external_population_size=external_population_size
        self.k=int(np.sqrt(self.population_size+self.external_population_size))
    #def update_archive(self):
    #   pass


    def fitness(self,objects):
        return self.raw_fitness(objects)+self.density_information(objects)

    def raw_fitness(self,objects):
        rankings=self.non_dominance_ranking(objects)
        raw_fitness=list()
        for i in range(len(rankings)):
            raw_fitness.append(len(rankings[rankings<rankings[i]]))
        raw_fitness=np.array(raw_fitness)
        return raw_fitness

    def density_information(self,objects):
        distances=self.kth_nearest_neighbor(objects,k=self.k)
        return 1/(distances+2)

    def kth_nearest_neighbor(self,objects,k=None):
        if k==None:
            k=self.k
        else:
            pass
        neigh=NearestNeighbors(n_neighbors=k,algorithm="ball_tree").fit(objects)
        distances, indices=neigh.kneighbors(objects)
        return distances[:,k-1]


    def combine(self,population1,population2):
        actual_size1=len(population1.chromosomes)
        actual_size2=len(population2.chromosomes)
        overall_population=Population(
            population_size=actual_size1+actual_size2,empty=True)
        for i in range(actual_size1):
            overall_population.chromosomes.append(population1.chromosomes[i])
        for i in range(actual_size2):
            overall_population.chromosomes.append(population2.chromosomes[i])
        return overall_population

    def environmental_selection(self,fitness,objects,population_size):#this function is not completed!!!!
        selected=np.where(fitness<1)
        objects_copy=copy.copy(objects)
        objects_copy=objects_copy[selected]

        if len(selected)==population_size:
            pass
        elif len(selected)>population_size:
            #if nondominated solutions are excessive

            # My completion is not the same on PPT.
            # To simplify, I remove solution with the min 1st nearest distance in every iteration.
            while len(selected)>population_size:
                distances=self.kth_nearest_neighbor(objects_copy,k=2)
                removed=distances.argmin()
                selected=np.delete(selected,removed,axis=0)
                objects_copy=np.delete(objects_copy,removed,axis=0)
        else:
            #if nondominated solutions are insufficient
            seq=np.argsort(fitness)
            selected=seq[0:population_size]

        return selected

    def binary_tournament(self,selected,fitness,size,number_of_candidates):
        #caution: this function return the index of selected, meaning that they directly refer to original population
        parents=list()
        while len(parents)<size:
            candidate1, candidate2=np.random.choice(number_of_candidates, size=2, replace=False)
            if fitness[candidate1]>fitness[candidate2]:
                parents.append(candidate2)
            else:
                parents.append(candidate1)

        parents=np.array(parents)
        return parents


    def spea2(self,t=20):
        parent_population=Population(population_size=population_size)
        external_population=Population(population_size=external_population_size,empty=True)
        system=AHPsystem()
        average_cost=list()
        average_minus_availability=list()
        min_cost=list()
        min_minus_availability=list()

        for round in range(t):
            #caution: we use (-availability)
            #combine Pt and Et
#print(round)
            overall_population=self.combine(parent_population,external_population)
            objects=np.empty(shape=(overall_population.size, 2))

            #calculate the object function value of all solutions in parent and external
            for i in range(objects.shape[0]):
                objects[i, 0]=overall_population.chromosomes[i].get_cost(system)
                objects[i, 1]=(-1)*overall_population.chromosomes[i].get_availability(system)

            #track the cost and availability
            average_cost.append(np.mean(objects[:, 0]))
            average_minus_availability.append(np.mean(objects[:, 1]))
            min_cost.append(np.min(objects[:, 0]))
            min_minus_availability.append(np.min(objects[:, 1]))

            #calculate fitness
            fitness=self.fitness(objects)

            #environmental selection
            selected=self.environmental_selection(fitness,objects,external_population.size)# return an array of the index in total_population
            new_external_population=Population(population_size=external_population_size,empty=True)
            if len(selected)!=new_external_population.size:
                print("error")
                return
            for i in range(len(selected)):
                new_external_population.chromosomes.append(overall_population.chromosomes[selected[i]])

            fitness_selected=fitness[selected]
            '''
            from_parents=selected[selected<parent_population.size]
            from_external=selected[selected>=parent_population.size]-parent_population.size

            fitness_parents=fitness[0:parent_population.size]
            fitness_external=fitness[parent_population.size:]
            fitness_selected=np.concatenate(
                (fitness_parents[from_parents],fitness_external[from_external]))
            
            #record the selected fitness
            for i in range(len(from_parents)):
                new_external_population.chromosomes.append(parent_population.chromosomes[int(from_parents[i])])
                #fitness_selected.append(fitness[from_parents[i]])
            for i in range(len(from_external)):
                new_external_population.chromosomes.append(external_population.chromosomes[int(from_external[i])])
                #fitness_selected.append(fitness[from_external[i]])
                
            '''
            #select out parents from Et+1
#fitness_selected=fitness[selected]
            parents=self.binary_tournament(fitness=fitness_selected,selected=selected,size=population_size,number_of_candidates=new_external_population.size)

            #crossover
            new_parent_population=Population(population_size,empty=True)
            for i in range(0, len(parents), 2):
                new_code1, new_code2=new_external_population.chromosomes[parents[i]].crossover(
                    new_external_population.chromosomes[parents[i+1]]
                )
                new_parent_population.chromosomes.append(Chromosome(code=new_code1))
                new_parent_population.chromosomes.append(Chromosome(code=new_code2))
            if(len(new_parent_population.chromosomes)!=new_parent_population.size):
                print("error")
                return

            #mutation
            for i in range(new_parent_population.size):
                new_parent_population.chromosomes[i].mutation()

            #update
            parent_population=new_parent_population
            external_population=new_external_population

        #return measures and solutions
        solutions=self.final_solutions(external_population,system)
        return [average_cost,average_minus_availability,min_cost,min_minus_availability,solutions]

#%%
#main body
if __name__ == '__main__':
    a=NSGA2()
    average_cost,average_minus_availability,min_cost,min_minus_availability,solutions=a.nsga2(t=100)
    #for plot
    plt.plot(min_cost,color="red")
    plt.plot(average_cost, color="blue")
    plt.ylim(min(min_cost),max(min_cost))
    plt.show()

    b=SPEA2()
    average_cost_2, average_minus_availability_2, min_cost_2, min_minus_availability_2, solutions_2=b.spea2(t=100)
    plt.plot(min_cost_2, color="red")
    plt.plot(average_cost_2, color="blue")
    plt.ylim(min(min_cost_2), max(min_cost_2))
    plt.show()
#%% test area for homework
x=np.array([[2,3],[4,5],[5,3],[4.5,1],[0.5,5],[2.5,0.5],[4.6,4.0],[3.0,4]])
n=np.arange(8)
'''
class Algorithm:
    def __init__(self,object_number=2,population_size=100):
        self.object_number=object_number
        self.population_size=population_size
        self.initialize()

    def initialize(self):
        self.Population=Population(self.population_size)

    def is_non_dominated(self,objects):#default minimization
        #objects is of shape(solutions,2) 2 for cost,(-availability)

        #the code below has referred to https://stackoverflow.com/a/40239615
        is_non_dominated=np.ones(objects.shape[0], dtype = bool)
        for i, c in enumerate(objects):
            if is_non_dominated[i]:
                is_non_dominated[is_non_dominated]=np.any(objects[is_non_dominated]<c, axis=1)  # Keep any point with a lower cost
                is_non_dominated[i]=True  # And keep self
        return is_non_dominated
'''
a=NSGA2()
a.is_non_dominated(x)

fig,ax=plt.subplots()
ax.scatter(x[:,0],x[:,1],c='r')


for i,txt in enumerate(n):
    ax.annotate(txt,x[i])

plt.show()
'''
    def crowding_distance(self,objects):
        objects_copy=objects
        number_of_objects=objects_copy.shape[-1]
        distances=np.zeros(len(objects_copy),dtype="float")
        for i in range(number_of_objects):
            distances+=self.crowding_distance_one_object(objects_copy[:,i])

        return distances
'''
a.crowding_distance(x)
#%%
#test area2
'''
a=NSGA2()
x=np.array([[2. , 3. ],
       [4. , 5. ],
       [5. , 3. ],
       [4.5, 1. ],
       [0.5, 5. ],
       [2.5, 0.5],
       [4.6, 4. ],
       [3. , 4. ]])

a.non_dominance_ranking(x)

#test area3
'''

'''
origin=np.empty(len(seq))
for i in range(len(seq)):
    origin[seq[i]]=aaa[i]
  
def re():
    return [1,2]

for i in range(0,10,2):
    print(i)
'''