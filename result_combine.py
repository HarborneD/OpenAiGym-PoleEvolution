import os
import sys

def combine_result_files(test_name):
	result_path = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])),"results")

	file_list = os.listdir(result_path)
	print("")
	print("file_list")
	print(file_list)

	output_file_path = os.path.join(result_path,test_name+".csv")
	
	with open(output_file_path,"w") as output_file:
		output_file.write(str("experiment_id,mutation_chance,Mutation_degree_limit,generation,best,worst,mean"))

		for result_file in file_list:
			print("")
			print("result_file")
			print(result_file)

			if result_file.startswith(test_name):
				file_path = os.path.join(result_path,result_file)
				with open(file_path,"r") as result_file:
					for line in result_file.readlines():
						if len(line) > 0:
							output_file.write(line)


combine_result_files("evolve_tests_highRes")