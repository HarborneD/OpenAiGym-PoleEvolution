import sys
import os
sys.path.append("C:\\Users\\Dan\\Dropbox\\Projects\\PyNeuralNet")
from PyNet import NeuralNetwork
import numpy as np
import gym
import operator
import math
import random
from time import gmtime, strftime
from multiprocessing import Pool, TimeoutError



#create 1000 neuralnetwork weights
def create_random_gen_1_agents(num_agents,init_weight_config,path_for_weights="",experiment_id="0"):
    weight_nn = NeuralNetwork()
    if(path_for_weights == ""):
        path_for_weights = os.path.dirname(os.path.realpath(sys.argv[0]))+ "\\weights"

    save_path = path_for_weights + "\\" + str(experiment_id)+"\\gen_0\\"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for agent_num in range(0,num_agents):
        weight_nn.initialise_weights(init_weight_config,save_path+str(agent_num)+".xml")




#run the simulation for each and record the average timesteps
def run_simulation_for_balancer(agent_network,env=""):
    if env == "":
        env = gym.make('CartPole-v0')
    else:
        env.reset()

    trials = []

    

    for i_episode in range(20):
        observation = env.reset()
        for t in range(200):
            env.render(close=True)
            #print(observation)
            action = get_action_of_balancer(agent_network,np.matrix(observation))
            observation, reward, done, info = env.step(action)
            
            if(t == 199):
                trials.append(t+1)
                #print("Episode finished at maximum {} timesteps".format(t+1))
                break

            if done:
                trials.append(t+1)
                #print("Episode finished after {} timesteps".format(t+1))
                break
  

    return sum(trials)/len(trials)

def get_agent_network(weight_path,experiment,generation,agent_number):

    agent_path = weight_path+"\\"+str(experiment)+"\\gen_"+str(generation)+"\\"+str(agent_number)+".xml"

    return NeuralNetwork(weights_data_path = agent_path)


def get_action_of_balancer(neural_net,input_data):

    neural_net.input_data = input_data

    neural_net.run_net()

    action_predictions = neural_net.output_data[0].tolist()[0]

    return action_predictions.index(max(action_predictions))



#cull the lowest 500

#reproduce

def breed_asexually(weight_path,experiment,previous_generation_number,sorted_prev_gen_performances,elite_populous_ratio=0.5,mutation_chance=0.1,mutation_degree_limit=0.3,two_offspring=True):
    old_weight_path = weight_path + "\\" + str(experiment)+"\\gen_"+str(previous_generation_number)+"\\"
    new_gen_save_path = weight_path + "\\" + str(experiment)+"\\gen_"+str(previous_generation_number+1)+"\\"

    if not os.path.exists(new_gen_save_path):
        os.makedirs(new_gen_save_path)

    elite_agent_count = math.floor(len(sorted_prev_gen_performances) * elite_populous_ratio)

    if(elite_agent_count % 2 != 0):
        elite_agent_count +=1

    elite_agents = sorted_prev_gen_performances[:elite_agent_count]

    agent_count = -1

    nn_save_weights = NeuralNetwork()


    for agent in elite_agents:


        agent_1_path = old_weight_path+str(agent[0])+".xml"
        nn_save_weights.set_weights_from_file(agent_1_path)

        agent_count +=1
        agent_1_new_path = new_gen_save_path+str(agent_count)+".xml"
        nn_save_weights.layer_weights_to_file(agent_1_new_path)

        agent_1_weights = nn_save_weights.layer_weights

        new_weights = []
        
        for layer_count in range(0,len(agent_1_weights)):
            layer_list=[]

            agent_1_layer = agent_1_weights[layer_count].tolist()
            
            for node_count in range(0,len(agent_1_layer)):
                node_list = []
                for weight_count in range(0,len(agent_1_layer[node_count])):
                    node_list.append(agent_1_layer[node_count][weight_count])
                    
                    if(random.random() <= mutation_chance):
                        node_list[-1] += (1 + (-2 * random.randrange(0,2)) ) * random.uniform(0,mutation_degree_limit)
                layer_list.append(node_list)
            
            new_weights.append(np.matrix(layer_list))
    
        nn_save_weights.layer_weights = new_weights

        agent_count +=1
        offspring_path = new_gen_save_path+str(agent_count)+".xml"
        nn_save_weights.layer_weights_to_file(offspring_path)


def breed_generation(weight_path,experiment,previous_generation_number,sorted_prev_gen_performances,elite_populous_ratio=0.5,mutation_chance=0.1,mutation_degree_limit=0.3,two_offspring=True):
    old_weight_path = weight_path + "\\" + str(experiment)+"\\gen_"+str(previous_generation_number)+"\\"
    new_gen_save_path = weight_path + "\\" + str(experiment)+"\\gen_"+str(previous_generation_number+1)+"\\"

    if not os.path.exists(new_gen_save_path):
        os.makedirs(new_gen_save_path)

    elite_agent_count = math.floor(len(sorted_prev_gen_performances) * elite_populous_ratio)

    if(elite_agent_count % 2 != 0):
        elite_agent_count +=1

    elite_agents = sorted_prev_gen_performances[:elite_agent_count]

    breeding_pairs = []

    nn_save_weights = NeuralNetwork()
    while(len(elite_agents) > 0):
        agent_1_index = random.randrange(0,len(elite_agents))
        agent_1 = elite_agents.pop(agent_1_index)
        
        agent_2_index = random.randrange(0,len(elite_agents))
        agent_2 = elite_agents.pop(agent_2_index)
        
        breeding_pairs.append((agent_1,agent_2))
    
    #first offspring
    agent_count = -1
    for pair in breeding_pairs:


        agent_1_path = old_weight_path+str(pair[0][0])+".xml"
        nn_save_weights.set_weights_from_file(agent_1_path)

        agent_count +=1
        agent_1_new_path = new_gen_save_path+str(agent_count)+".xml"
        nn_save_weights.layer_weights_to_file(agent_1_new_path)

        agent_2_path = old_weight_path+str(pair[1][0])+".xml"
        nn_save_weights.set_weights_from_file(agent_1_path)

        agent_count +=1
        agent_2_new_path = new_gen_save_path+str(agent_count)+".xml"
        nn_save_weights.layer_weights_to_file(agent_2_new_path)


        nn_save_weights.layer_weights = breed_agent_pair(weight_path,experiment,previous_generation_number,agent_1_path,agent_2_path,"node",mutation_chance,mutation_degree_limit)
        
        agent_count +=1
        offspring_path = new_gen_save_path+str(agent_count)+".xml"
        nn_save_weights.layer_weights_to_file(offspring_path)

    if(two_offspring):
        second_pairs_list = []
        for new_pair_count in range(0,len(breeding_pairs)):
            agent_1 = breeding_pairs[new_pair_count][0]
            
            agent_2 = breeding_pairs[-1 -new_pair_count][1]
            
            second_pairs_list.append((agent_1,agent_2))

        for pair in second_pairs_list:
            agent_1_path = old_weight_path+str(pair[0][0])+".xml"
            
            agent_2_path = old_weight_path+str(pair[1][0])+".xml"

            nn_save_weights.layer_weights = breed_agent_pair(weight_path,experiment,previous_generation_number,agent_1_path,agent_2_path,"node",mutation_chance,mutation_degree_limit)
            
            agent_count +=1
            offspring_path = new_gen_save_path+str(agent_count)+".xml"
            nn_save_weights.layer_weights_to_file(offspring_path)

def breed_agent_pair(weight_path,experiment,generation_number,agent_1_path,agent_2_path,breed_type="node",mutation_chance=0.1,mutation_degree_limit=0.3):
    weights_nn = NeuralNetwork()

    
    agent_1_weights = weights_nn.get_weights_from_file(agent_1_path)

    agent_2_weights = weights_nn.get_weights_from_file(agent_1_path)

    new_weights = []


    if(breed_type == "node"):
        for layer_count in range(0,len(agent_1_weights)):
            layer_list=[]

            agent_1_layer = agent_1_weights[layer_count].tolist()
            
            agent_2_layer = agent_1_weights[layer_count].tolist()


            for node_count in range(0,len(agent_1_layer)):
                node_list = []
                for weight_count in range(0,len(agent_1_layer[node_count])):
                    if(random.randrange(0,2) == 1):
                        node_list.append(agent_1_layer[node_count][weight_count])
                    else:
                        node_list.append(agent_2_layer[node_count][weight_count])

                    if(random.random() <= mutation_chance):
                        node_list[-1] += (1 + (-2 * random.randrange(0,2)) ) * random.uniform(0,mutation_degree_limit)
                layer_list.append(node_list)
            
            new_weights.append(np.matrix(layer_list))
    
    return new_weights


def run_evolutionary_learn_experiment(experiment_id,path_for_weights,mutation_chance=0.1,mutation_degree_limit=0.3,agents_per_generation=100,number_of_generations=20,breed_with_pairs=True):

    create_random_gen_1_agents(agents_per_generation,[4,2,2],path_for_weights,experiment_id)
    generation_results = []

    env = gym.make('CartPole-v0')

    for generation in range(0,number_of_generations):
        print("Generation: "+str(generation))
        performances = []

        for agent_num in range(0,agents_per_generation):

            net = get_agent_network(path_for_weights,experiment_id,generation,agent_num)
            performances.append((agent_num,run_simulation_for_balancer(net,env)))
       

        sorted_performances = sorted(performances, key=lambda x: x[1], reverse=True)
        print(sorted_performances)
   
        generation_result = ( ("Best",sorted_performances[0]),("Worst",sorted_performances[-1]), ("Mean",sum([x[1] for x in sorted_performances])/len(sorted_performances)) )
        
        generation_results.append(("gen "+str(generation),generation_result))
        
        if(generation < (number_of_generations-1)):
            if(breed_with_pairs):
                breed_generation(path_for_weights,experiment_id,generation,sorted_performances)
            else:
                breed_asexually(path_for_weights,experiment_id,generation,sorted_performances)

    return generation_results
   


def run_evolution_trials(path_for_weights="",agents_per_generation=100,number_of_generations=20,breed_with_pairs=True,mutation_chance_start=0.05,mutation_chance_limit=0.5,mutation_chance_step=0.05,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.5,mutation_degree_limit_step=0.05,reps_per_type=1):

    if path_for_weights == "":
        path_for_weights = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"weights")

    experiment_results = []

    for mutation_chance in np.arange(mutation_chance_start,mutation_chance_limit,mutation_chance_step):
        for mutation_degree_limit in np.arange(mutation_degree_limit_start,mutation_degree_limit_limit,mutation_degree_limit_step):
            for repeatition in range(0,reps_per_type):
                experiment_id = "MC_" + str(mutation_chance/0.05) + "-MDL_" + str(mutation_degree_limit/0.05) + "-REP_"+str(repeatition+1)

                experiment_label = (experiment_id,mutation_chance,mutation_degree_limit)

                experiment_results.append( (experiment_label,run_evolutionary_learn_experiment(experiment_id,path_for_weights,mutation_chance,mutation_degree_limit,agents_per_generation,number_of_generations,breed_with_pairs) ) ) 

    return experiment_results


def parallel_support_func(args):
    experiment_results = []

    experiment_label = args[0]
    experiment_id = args[1]
    path_for_weights = args[2]
    mutation_chance = args[3]
    mutation_degree_limit = args[4]
    agents_per_generation = args[5]
    number_of_generations = args[6]
    breed_with_pairs = args[7]
    trial_name = args[8]

    experiment_results.append( (experiment_label,run_evolutionary_learn_experiment(experiment_id,path_for_weights,mutation_chance,mutation_degree_limit,agents_per_generation,number_of_generations,breed_with_pairs) ) ) 

    if(trial_name == ""):
        trial_name = "trial " + str(experiment_label)
    else:
        trial_name += (" " + str(experiment_label))
    output_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"results")

    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    file_path = os.path.join(output_path,trial_name+".csv")

    output_trial_results(experiment_results,trial_name,output_path,"a")


def run_evolution_trials_parallel(path_for_weights="",agents_per_generation=100,number_of_generations=20,breed_with_pairs=True,mutation_chance_start=0.05,mutation_chance_limit=0.5,mutation_chance_step=0.05,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.5,mutation_degree_limit_step=0.05,reps_per_type=1,trial_name=""):

    if path_for_weights == "":
        path_for_weights = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"weights")


    experiment_args = []

    for mutation_chance in np.arange(mutation_chance_start,mutation_chance_limit,mutation_chance_step):
        for mutation_degree_limit in np.arange(mutation_degree_limit_start,mutation_degree_limit_limit,mutation_degree_limit_step):
            for repeatition in range(0,reps_per_type):
                experiment_id = "MC_" + str(mutation_chance/0.05) + "-MDL_" + str(mutation_degree_limit/0.05) + "-REP_"+str(repeatition+1)

                experiment_file_name = "evolve_tests_highRes ('" +"MC_" + str(round(mutation_chance/0.05,3)) + "-MDL_" + str(round(mutation_degree_limit/0.05,3)) + "-REP_"+str(repeatition+1)+"', '"+str(mutation_chance)+"', '"+str(mutation_degree_limit)+"').csv"
                
                check_path = os.path.join("results",experiment_file_name)
                
                if(os.path.exists(check_path)):
                    continue
                
                experiment_label = (experiment_id,str(round(mutation_chance,3)),str(round(mutation_degree_limit,3)))

                experiment_args.append( (experiment_label,experiment_id,path_for_weights,mutation_chance,mutation_degree_limit,agents_per_generation,number_of_generations,breed_with_pairs,trial_name)  )

    with Pool() as pool:
        pool.map(parallel_support_func,experiment_args)

    


def run_evolution_trials_outputoften(path_for_weights="",agents_per_generation=100,number_of_generations=20,breed_with_pairs=True,mutation_chance_start=0.05,mutation_chance_limit=0.5,mutation_chance_step=0.05,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.5,mutation_degree_limit_step=0.05):

    if path_for_weights == "":
        path_for_weights = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"weights")

    trial_name = "trial " + strftime("%Y-%m-%d %H-%M-%S", gmtime())

    output_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"results")

    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    file_path = os.path.join(output_path,trial_name+".csv")

    with open(str(file_path),"w") as output_file:
        output_file.write(str("experiment_id,mutation_chance,Mutation_degree_limit,generation,best,worst,mean"))


    for mutation_chance in np.arange(mutation_chance_start,mutation_chance_limit,mutation_chance_step):
        for mutation_degree_limit in np.arange(mutation_degree_limit_start,mutation_degree_limit_limit,mutation_degree_limit_step):
            experiment_results = []

            experiment_id = 1000 * (mutation_chance/0.05) + (mutation_degree_limit/0.05)

            experiment_label = (experiment_id,round(mutation_chance,3),round(mutation_degree_limit,3))
            print("Starting: "+str(experiment_label))
            experiment_results.append( (experiment_label,run_evolutionary_learn_experiment(experiment_id,path_for_weights,mutation_chance,mutation_degree_limit,agents_per_generation,number_of_generations,breed_with_pairs) ) ) 

            output_trial_results(experiment_results,trial_name,output_path,"a")
    


def output_trial_results(trial_results,trial_name="",output_path="",write_type="w"):
    if trial_name == "":
        trial_name = "trial " + strftime("%Y-%m-%d %H-%M-%S", gmtime())

    if output_path == "":
        output_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"results")

    if(not os.path.exists(output_path)):
        os.makedirs(output_path)

    file_path = os.path.join(output_path,trial_name+".csv")

    output_string =""

    with open(str(file_path),write_type) as output_file:
        if(write_type != "a"):
            output_string += str("experiment_id,mutation_chance,Mutation_degree_limit,generation,best,worst,mean")

        for experiment in trial_results:
            label = experiment[0]

            for result in experiment[1]:
                output_string += str("\n")

                output_string += str(str(label[0])+",")
                output_string += str(str(label[1])+",")
                output_string += str(str(label[2])+",")

                gen_number = result[0].replace("gen ","")
                output_string += (str(gen_number)+",")

                best_tuple = result[1][0][1]
                output_string += (str(best_tuple[1])+",")

                worst_tuple = result[1][1][1]
                output_string += (str(worst_tuple[1])+",")

                mean = result[1][2][1]
                output_string += str(mean)


        output_file.write(output_string)



#output_trial_results(run_evolution_trials(number_of_generations=20,mutation_chance_start=0.05,mutation_chance_limit=0.5,mutation_chance_step=0.05,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.5,mutation_degree_limit_step=0.05))

#run_evolution_trials_outputoften(number_of_generations=20,mutation_chance_start=0.05,mutation_chance_limit=0.5,mutation_chance_step=0.05,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.5,mutation_degree_limit_step=0.05)
if __name__ == '__main__':
    run_evolution_trials_parallel(number_of_generations=20,mutation_chance_start=0.2,mutation_chance_limit=0.7,mutation_chance_step=0.025,mutation_degree_limit_start=0.05,mutation_degree_limit_limit=0.7,mutation_degree_limit_step=0.025,reps_per_type=5,trial_name="evolve_tests_highRes")

