# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 16:45:42 2022

@author: adeel
"""

import os
import pandas as pd
import glob


def evaluate(transformation, generated_path, path):
	accuracy_list = []
	overall_correct_cnt = 0
	overall_occurences_cnt = 0
	overall_accuracy = 1.0
	accuracy = 1.0
	for generatedFile in generated_dict:
		print ("\nProcessing: "+generatedFile)
		with open(path+"/"+generatedFile) as fp_original, open(generated_path+"/"+generatedFile) as fp_generated:
			generated = fp_generated.readlines()
			original = fp_original.readlines()
			occurrences_cnt = 0
			correct_cnt = 0
			for i, line in enumerate(original):
				for j, char in enumerate(line):
					for key in transformation_dict[transformation]:
						if char==key:
							occurrences_cnt += 1
							overall_occurences_cnt += 1
							print('Replaced Block: '+original[i][j]+ '\tPredicted Block:'+generated[i][j])
							if generated[i][j] == original[i][j]:
								correct_cnt += 1
								overall_correct_cnt += 1
			if occurrences_cnt != 0:
				accuracy = correct_cnt/occurrences_cnt
				accuracy_list.append(accuracy)
				print ("Prediction Accuracy for Level: ", accuracy)
			else:
				accuracy_list.append(accuracy)
	if overall_occurences_cnt != 0:
		overall_accuracy = overall_correct_cnt/overall_occurences_cnt
		print ("\nPrediction Accuracy for Set: ", overall_accuracy)
	return(accuracy_list, overall_accuracy)

def evaluate_window(transformation, generated_path, path, window_size):
	accuracy_list = []
	overall_correct_cnt = 0
	overall_occurences_cnt = 0
	overall_accuracy = 1.0
	accuracy = 1.0
	for generatedFile in generated_dict:
		with open(path+"/"+generatedFile) as fp_original, open(generated_path+"/"+generatedFile) as fp_generated:
			generated = fp_generated.readlines()
			original = fp_original.readlines()
			occurrences_cnt = 0
			correct_cnt = 0
			for i, line in enumerate(original):
				for j, char in enumerate(line):
					for key in transformation_dict[transformation]:
						if char==key:
							print ("\nProcessing: "+generatedFile)
							occurrences_cnt += 1
							overall_occurences_cnt += 1
							for m in range(0,window_size):
								for n in range (0,window_size):
									try:
										if generated[i][j] == original[i][j]:
											correct_cnt += 1
											overall_correct_cnt += 1
											print('Replaced Block: '+original[i][j]+'\t at '+str(i)+','+str(j)+ '\tPredicted Block:'+generated[i-m][j-n]+'\t at '+str(i)+','+str(j))
											break;
										elif generated[i-m][j-n] == original[i][j]:
											correct_cnt += 1
											overall_correct_cnt += 1
											print('Replaced Block: '+original[i][j]+'\t at '+str(i)+','+str(j)+ '\tPredicted Block:'+generated[i-m][j-n]+'\t at '+str(i-m)+','+str(j-n))
											break;
										elif generated[i+m][j+n] == original[i][j]:
											correct_cnt += 1
											overall_correct_cnt += 1
											print('Replaced Block: '+original[i][j]+'\t at '+str(i)+','+str(j)+ '\tPredicted Block:'+generated[i+m][j+n]+'\t at '+str(i+m)+','+str(j+n))
											break;
										else:
											pass
									except IndexError:
										continue
								break;
			if occurrences_cnt != 0:
				accuracy = correct_cnt/occurrences_cnt
				accuracy_list.append(accuracy)
				print ("Prediction Accuracy for Level: ", accuracy)
			else:
				accuracy_list.append(accuracy)
	if overall_occurences_cnt != 0:
		overall_accuracy = overall_correct_cnt/overall_occurences_cnt
		print ("\nPrediction Accuracy for Set: ", overall_accuracy)
	return(accuracy_list, overall_accuracy)

def evaluate_anahita(margins, modified_chars, original_str, path):
  correct_char_count = 0
  total_modified_char_count = 0
  output_files = glob.glob(path+'/*.txt')
  # print("\noriginal is")
  # print(original_str)
  original_str = original_str.splitlines()

  for path in output_files:
    # print(path)
    f = open(path)
    output_str = f.read()
    # print("\nPredicted is")
    # print(output_str)
    output_str = output_str.splitlines()
    out_shape = (len(output_str), len(output_str[0]))
    m, n = margins

    for i, line in enumerate(original_str):
      for j, char in enumerate(line):
        if char in modified_chars:
          window = output_str[max(0, i-m): min(out_shape[0], i+m+1)][max(0, j-n): min(out_shape[1], j+n+1)]
          if any(any(e in l for e in modified_chars) for l in window):
            correct_char_count += 1
          total_modified_char_count +=1

  if total_modified_char_count:
    return correct_char_count/total_modified_char_count
  else:
    return 0


#path: str = './data/basic_13_tokens'
path = r'./data/Desired Update'
# ana_path: str = os.getcwd()+'\SMB1_Data\Processed\mario-1-1.txt'
# f = open(ana_path)
# output_str = f.read()

generated_path = r'./data/Update'

transformation_dict = {
        'q_mark': {'Q':'S','?':'S'},
        'cannon': {'b':'E','B':'E'},
        'coin': {'o':'-'},
    }

window_size = 1

accuracy_list_of_lists = []
overall_accuracy_list = []

# for m, n in [(1,1), (2,2), (2,3), (3,3), (5,5), (10,10)]:  
#   print("===========Accuracy results using a {}x{} window===========".format(2*m+1, 2*n+1)) 
#   print("Accuracy of coin levels is: ", evaluate_anahita((m, n),['B','b'], output_str, generated_path))
#   print("=========================================================\n")


for key in transformation_dict:
	path_key = path+fr'/{key}'
	generated_path_key = generated_path+fr'/{key}'

	original_dict = os.listdir(path_key)
	generated_dict = os.listdir(generated_path_key)

	accuracy_list, overall_accuracy = evaluate_window(key, generated_path_key, path_key, window_size)
	overall_accuracy_list.append(overall_accuracy)
	accuracy_list_of_lists.append(accuracy_list)
    
print(overall_accuracy_list)