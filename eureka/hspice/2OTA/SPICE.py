import os
import shutil
import sys
import subprocess as sp
import time
import numpy as np
import json
import argparse
import gymnax



def Measure(x, identification = 0):
	current_directory = os.path.dirname(os.path.abspath(__file__))
	print('current directory: ', current_directory)
	test = 'amp_test_180nm' + '.sp'
	i = open(os.path.join(current_directory, test), "r")
	ilines = i.readlines()
	len_ilines = len(ilines)
	test_modified = 'amp_test_180nm_modified' + str(identification) + '.sp'
	j = open(os.path.join(current_directory, test_modified), "w")

	for xi in range(len(x)):
		x[xi] = str(x[xi])
	for line_num in range(len_ilines):
		if '.PARAM' in ilines[line_num]:
			j.write(".PARAM cf=%s l1=%s l2=%s l3=%s l4=%s l5=%s " %(x[0], x[1], x[2], x[3], x[4], x[5]))
			j.write("mcap=%s n1=%s n2=%s n3=%s " %(x[6], x[7], x[8], x[9]))
			j.write("w1=%s w2=%s w3=%s w4=%s w5=%s res=%s\n" %(x[10], x[11], x[12], x[13], x[14], x[15])) 
		else:
			j.write(ilines[line_num])
	i.close()
	j.close()
	check_license = 1
	no_converge = 0
	while check_license:
		sim_cmd = 'hspice -i ' + os.path.join(current_directory, test_modified) + ' -o output' + str(identification) #amp_test_180nm_modified' + str(identification) + '.sp -o output' + str(identification)
		sp.call(sim_cmd, shell=True)
		check_lis = open(os.path.join(current_directory, 'output' + str(identification) +'.lis'), "r")
		check_lis_lines = check_lis.readlines()
		for line_num in range(len(check_lis_lines)):
			if 'License server machine is down or not responding.' in check_lis_lines[line_num]:
				check_license = 1
				break
			elif 'Unable to checkout' in check_lis_lines[line_num]:
				check_license = 1
				break
			else:
				check_license = 0
				if 'no convergence in operating point' in check_lis_lines[line_num]:
					no_converge = 1
					break
		check_lis.close()
	if no_converge:
		out = np.zeros(9)
		out[0] = 0
		out[1] = -180
		out[2] = 1e-6
		out[3] = 0
		out[4] = 0
		out[5] = 0
		out[6] = 0
		out[7] = 1
		out[8] = 1
	else:
		f = open(os.path.join(current_directory, 'output' + str(identification) + '.lis'), "r")
		g = open(os.path.join(current_directory, 'results_180nm_dum' + str(identification) + '.txt'), "w")
		temp=0
		flines = f.readlines()
		for line_num in range(len(flines)):
			temp_line = flines[line_num].split(" ")
			for i in range(len(temp_line)):
				if 'ugf=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'pm=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'dc_gain=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'cmrr=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'psrr=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'os=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'upper_trig=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif 'lower_trig=' in temp_line[i]:
					g.write(flines[line_num])
					break
				elif temp_line[i]=='volts':
					g.write(flines[line_num]) 
					break
				elif 'pwr=' in temp_line[i]:
					g.write(flines[line_num])
					break
		f.close()
		g.close()
		f = open(os.path.join(current_directory, 'results_180nm_dum' + str(identification) + '.txt'), "r")
		g = open(os.path.join(current_directory, 'results_180nm' + str(identification) + '.txt'), "w")
		flines = f.readlines()
		for line_num in range(len(flines)):
			temp_line = flines[line_num].split("=")
			if 'upper_trig' in temp_line[0] :
				if len(temp_line) < 4:
					UPPER = "failed"
				else:
					UPPER = temp_line[2].replace(" ", "").split(" ")[0].replace("trig", "")
			elif 'lower_trig' in temp_line[0] :
				if len(temp_line) < 4:
					LOWER = "failed"
				else:
					LOWER = temp_line[2].replace(" ", "").split(" ")[0].replace("trig", "")
			else:
				g.write(temp_line[1].replace("volts", "").replace(" ", ""))
		if ("failed" in UPPER) and ("failed" in LOWER) :
			g.write("failed\n")
		elif ("failed" in UPPER) :
			g.write(LOWER + '\n')
		elif ("failed" in LOWER) :
			g.write(UPPER + '\n')
		else :
			if ("notfound" in UPPER) or ("notfound" in LOWER):
				g.write("failed\n")	
			elif float(UPPER) > float(LOWER):
				g.write(str(UPPER) + '\n')
			else:
				g.write(str(LOWER) + '\n')
		f.close()
		g.close()
		g = open(os.path.join(current_directory, 'results_180nm' + str(identification) + '.txt'), "r")
		glines = g.readlines()
		out = np.zeros(9)
		out[0] = 0 if "failed" in glines[1] else float(glines[1])
		if "failed" in glines[2]:
			out[1] = -180
		else:
			out[1]=convert_to_within_180_degrees(float(glines[2]))
		out[2] = 1e-6 if "failed" in glines[8] else float(glines[8])
		out[3] = glines[5]
		out[4] = glines[3]
		out[5] = glines[4]
		out[6] = glines[6]
		out[7] = glines[0]
		out[8] = glines[7]
		g.close()	
	return out  

def FoM(Measure_out, weights, constraints):
	len_constraints = len(constraints)
	print(len_constraints)


	#print(len_constraints);print(len_Measure_out);print(accum);print(Measure_out);print(Measure_out_t)
	#print(len_Measure_out) ### 9
	#print(len(Measure_out_t))
	#tf.print("Measure_out in FoM");tf.print(Measure_out);
	Measure_out = np.array(Measure_out, dtype=np.float32)
	#print("tf.shape(Measure_out) in FoM: "); print(tf.shape(Measure_out))
	accum = np.zeros(1) #accum = tf.zeros([len_Measure_out,1])

	#print(len(Measure_out_t)) ### 9
	#Measure_out_t_reshape = tf.reshape(Measure_out_t, [len_constraints, len_Measure_out, 1])
	print(constraints)
	print(weights)
	for i in range(len_constraints):
		if constraints[i][1]==1:
			accum += weights[i]*np.minimum(1.0,np.maximum(0.0,(Measure_out[i] - constraints[i][0])/constraints[i][0]))
		else:
			accum += weights[i]*np.minimum(1.0,np.maximum(0.0,(constraints[i][0] - Measure_out[i])/constraints[i][0]))
	FoM_out = Measure_out[len_constraints] + accum

	'''for i in range(len_constraints):
		accum += weights[i]*Measure_out[i]
	FoM_out = weights[len_constraints]*Measure_out[len_constraints] + accum'''

	return FoM_out

def convert_to_within_180_degrees(phase_margin):
    while phase_margin < -180:
        phase_margin += 360

    while phase_margin > 180:
        phase_margin -= 360

    return phase_margin


def main():
	parser = argparse.ArgumentParser(description="HSpice Simulation with FoM Calculation")
	parser.add_argument('--env_name', type=str, required=True, help='Name of the Gymnax environment')
	for i in range(16):
		parser.add_argument(f'--x{i}', type=float, required=True, help=f'Value for x{i}')

	args = parser.parse_args()

	# Create a NumPy array from the parsed arguments
	x = np.array([getattr(args, f'x{i}') for i in range(16)]) 

	# Get environment parameters based on env_name
	env, env_params = gymnax.make(args.env_name)
	constraints = {attr: getattr(env_params, attr) for attr in dir(env_params) if attr.endswith("_constraints")}
	ordered_constraints = sorted(constraints.items(), key=lambda item: int(item[0][3:-12]))  # Sort by numeric part of attribute name
	constraints_list = [list(value) for key, value in ordered_constraints]
	weights = env.weights
	FoM_input_constraints = constraints_list

	output = Measure(x)
	FoM_out = FoM(output, weights, FoM_input_constraints)

	# Save results to a JSON file
	results = {
		"env_name": args.env_name,
		"x": x.tolist(),
		"FoM": FoM_out.item(),
		"output": output.tolist()
	}
	with open("simulation_results.json", "w") as outfile:
		json.dump(results, outfile, indent=4)
	print(f"HSpice Simulation Results: {results}")
	print("Simulation completed and results saved to simulation_results.json")

if __name__ == "__main__":
	main()



