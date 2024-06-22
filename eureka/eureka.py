import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI

import re
import subprocess
from pathlib import Path
import shutil
import time 
import ast

from utils.misc import * 
from utils.file_utils import find_files_with_substring, load_tensorboard_logs
from utils.create_task import create_task
from utils.extract_task_code import *
from utils.custom_code_extract import parse_markdown_for_first_function_code

EUREKA_ROOT_DIR = (os.getcwd()) 
# current location is Eureka_Development/Eureka_Analog_Sizing/eureka while Eureka_Development/Eureka_Analog_Sizing is the root directory with the submodules that contain the environment code and the RL code
ISAAC_ROOT_DIR = f"{os.path.dirname(EUREKA_ROOT_DIR)}/submodules/gymnax_Analog_RL/gymnax/environments/custom" # f"{EUREKA_ROOT_DIR}/../isaacgymenvs/isaacgymenvs" # change to f"{EUREKA_ROOT_DIR}/../submodules/gymnax_Analog_RL/environments/custom"
PUREJAXRL_ROOT_DIR = f"{os.path.dirname(EUREKA_ROOT_DIR)}/submodules/purejaxrl_Analog_RL/purejaxrl"
HSPICE_ROOT_DIR = f"{EUREKA_ROOT_DIR}/hspice/2OTA" # Need to be changed for each design
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {EUREKA_ROOT_DIR}")


    task = cfg.env.task
    task_description = cfg.env.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Task: " + task)
    logging.info("Task description: " + task_description)

    env_name = cfg.env.env_name.lower()
    env_parent = 'custom' #'isaac' if f'{env_name}.py' in os.listdir(f'{EUREKA_ROOT_DIR}/envs/isaac') else 'dexterity' # add custom to env_parent
    task_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}.py'     # location of environment code
    task_obs_file = f'{EUREKA_ROOT_DIR}/envs/{env_parent}/{env_name}_obs.py' # location of environment observation code
    shutil.copy(task_obs_file, f"env_init_obs.py")
    task_code_string  = file_to_string(task_file)
    task_obs_code_string  = file_to_string(task_obs_file)
    output_file = f"{ISAAC_ROOT_DIR}/{env_name}{suffix.lower()}.py" # f"{ISAAC_ROOT_DIR}/tasks/{env_name}{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{EUREKA_ROOT_DIR}/utils/prompts'
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f'{prompt_dir}/code_feedback.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')
    policy_feedback = file_to_string(f'{prompt_dir}/policy_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(task_reward_signature_string=reward_signature) + code_output_tip
    initial_user = initial_user.format(task_obs_code_string=task_obs_code_string, task_description=task_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    task_code_string = task_code_string.replace(task, task+suffix)
    # Create Task YAML files
    create_task(ISAAC_ROOT_DIR, cfg.env.task, cfg.env.env_name, suffix)

    DUMMY_FAILURE = -10000.
    all_fitnesses = []
    best_fitnesses = []
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    best_fitness_overall = DUMMY_FAILURE
    max_reward_code_path = None 
    print("Debug 0: Initial System and User Messages")
    # Eureka generation loop
    for iter in range(cfg.iteration):
        # Get Eureka response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=cfg.temperature,
                        n=chunk_size
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens
        print('Debug 1: Eureka Response')
        if cfg.sample == 1:
            logging.info(f"Iteration {iter}: GPT Output:\n " + responses[0].message.content + "\n")

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = [] 
        rl_runs = []
        for response_id in range(cfg.sample):
            print('Debug 2: Generated Code Processing')
            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r'```python(.*?)```',
                r'```(.*?)```',
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])


            ## Experimental implementation that can extract nested functions
            # Replaces current parsing system
            # Remove the markdown code block delimiters
            code_string = None
            code_string = parse_markdown_for_first_function_code(response_cur)
            ## Experimental End


            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                logging.info(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                print('Debug Full Response:\n', response_cur)
                print('Debug Print Code String:\n', code_string)
                continue

            code_runs.append(code_string)
            reward_signature = [
                f"def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:\n",
                f"    reward, reward_component = compute_ota_reward(*(self.reward_compute_input(model_output, params)))\n",
                f"    return reward, reward_component\n",
            ]
            indent = " " * 4
            reward_signature = "".join([indent + line for line in reward_signature])
            reward_fun_pattern = r"(def compute_reward\(self,.*?\) -> float:.*?)(?=\n\s+def|\n\s+\Z)"
            # Replace the old compute_reward method with the new one
            task_code_string_iter = re.sub(reward_fun_pattern, reward_signature.strip(), task_code_string, flags=re.DOTALL)
            """
            if "def compute_reward(self)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in task_code_string:
                task_code_string_iter = task_code_string.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            elif "def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:" in task_code_string: # gymanx implementation unlike issacgym has functional programming elements so that it needs to receive model_output and params
                task_code_string_iter = task_code_string.replace("def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:", "def compute_reward(self, model_output: chex.Array, params: EnvParams) -> float:\n" + reward_signature)
            else:
                raise NotImplementedError
            """



            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n') # only Tuple used in compute_ota_reward function
                file.writelines("import jax.numpy as jnp" + '\n')
                # Rest of imports unnecessary for the reward function as it is from isaacgymenvs
                #file.writelines("from typing import Tuple, Dict" + '\n')
                #file.writelines("import math" + '\n')
                #file.writelines("import torch" + '\n')
                #file.writelines("from torch import Tensor" + '\n')
                # torch.jit not used as this environment uses jax instead, compute_ota_reward is jit compiled through the step_env function
                #if "@torch.jit.script" not in code_string:
                #    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            with open(f"env_iter{iter}_response{response_id}_rewardonly.py", 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"env_iter{iter}_response{response_id}.py")

            # Find the freest GPU to run GPU-accelerated RL
            set_freest_gpu()

            print('Debug 2.1: Running RL Training')            
            # Execute the python file with flags
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{PUREJAXRL_ROOT_DIR}/jax_train.py',  
                                            'hydra/output=subprocess',
                                            f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                            f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                            f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False',
                                            f'max_iterations={cfg.max_iterations}'],
                                            stdout=f, stderr=f)
            print(f'Debug print process: {process}')
            block_until_training(rl_filepath, log_status=True, iter_num=iter, response_id=response_id)
            rl_runs.append(process)
            print('Debug 2.2: RL Training Completed')
        print('Debug 3: Gathering RL Training Results and Constructing Reward Reflection')
        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
        code_paths = []
        fitness_scores = [] # Fitness scores from the Hspice simulation based on ouput gates
        
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            print("rl_run.communicate()")
            rl_run.communicate()
            rl_filepath = f"env_iter{iter}_response{response_id}.txt"
            code_paths.append(f"env_iter{iter}_response{response_id}.py")
            try:
                with open(rl_filepath, 'r') as f:
                    stdout_str = f.read() 
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                # Check if the tensorboard logdir exists
                if not os.path.exists(tensorboard_logdir) or not os.path.isdir(tensorboard_logdir):
                    pass
                else:                        
                    tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                    max_iterations = np.array(tensorboard_logs['gt_reward']).shape[0]
                    epoch_freq = max(int(max_iterations // 10), 1)
                    
                    content += policy_feedback.format(epoch_freq=epoch_freq)
                    
                    # Compute Correlation between Human-Engineered and GPT Rewards
                    if "gt_reward" in tensorboard_logs and "gpt_reward" in tensorboard_logs:
                        gt_reward = np.array(tensorboard_logs["gt_reward"])
                        gpt_reward = np.array(tensorboard_logs["gpt_reward"])
                        reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                        reward_correlations.append(reward_correlation)

                    # Add reward components log to the feedback
                    for metric in tensorboard_logs:
                        if "/" not in metric:
                            metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                            metric_cur_max = max(tensorboard_logs[metric])
                            metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                            if "consecutive_successes" == metric:
                                successes.append(metric_cur_max)
                            metric_cur_min = min(tensorboard_logs[metric])
                            if metric != "gt_reward" and metric != "gpt_reward":
                                if metric != "consecutive_successes":
                                    metric_name = metric 
                                else:
                                    metric_name = "task_score"
                                content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                            else:
                                # Provide ground-truth score when success rate not applicable
                                if "consecutive_successes" not in tensorboard_logs:
                                    content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    
                # Run Hspice simulation to get the fitness score to attach to content
                # extract x0~x15 values from env_iter*_response*.py
                print("Debug 4: Running Hspice Simulation")
                with open(f"env_iter{iter}_response{response_id}.txt", 'r') as f:
                    file_content = f.read()
                # Use a regular expression to extract the values of x0, x1, ..., x15
                pattern = r'x(\d+):\s*([-\d.e+]+)'
                matches = re.findall(pattern, file_content)
                # Convert the matches to a dictionary
                extracted_values = {f'x{match[0]}': float(match[1]) for match in matches}
                spice_script_args = ['python', '-u', f'{HSPICE_ROOT_DIR}/SPICE.py']
                for key, value in extracted_values.items():
                    spice_script_args.append(f"--{key}={value}")
                # Adding --env_name argument
                spice_script_args.append(f'--env_name={task}-custom')
                # Execute SPICE.py at HSPICE_ROOT_DIR using x and cfg.
                sim_filepath = f"env_hspice_sim_eval_iter{iter}_response{response_id}.txt"
                print(f"Debug 4-0: Hspice Simulation Started, Iteration {iter}, Response {response_id}, Command {spice_script_args}")
                with open(sim_filepath, 'w') as f:
                    process = subprocess.Popen(spice_script_args, stdout=f, stderr=f)
                block_until_sim(sim_filepath, log_status=True, iter_num=iter, response_id=response_id)
                print(f"Debug 4-1: Hspice Simulation Completed, Iteration {iter}, Response {response_id}")
                try:
                    with open(sim_filepath, 'r') as f:
                        sim_str = f.read()
                    content += stdout_str
                    # Using regular expression to extract the number following "Fom:"
                    match = re.search(r"'FoM': ([\d\.]+)", sim_str)
                    if match:
                        fom_number = float(match.group(1))
                    else:
                        fom_number = DUMMY_FAILURE
                except: 
                    content += "Hspice simulation failed to execute! Fitness score is not available!"
                    fom_number = DUMMY_FAILURE
                fitness_scores.append(fom_number)    

                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 
        
        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.sample != 1:
            execute_rates.append(0.)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logging.info("All code generation failed! Repeat this iteration from the current message checkpoint!")
            continue
        all_fitnesses.append(fitness_scores)
        # Select the best code sample based on the fitness score from hspice simulation#success rate
        #best_sample_idx = np.argmax(np.array(successes))
        best_sample_idx = np.argmin(np.array(fitness_scores))
        best_content = contents[best_sample_idx]
        
        best_fitness = fitness_scores[best_sample_idx]
        #max_success = successes[best_sample_idx]
        #max_success_reward_correlation = reward_correlations[best_sample_idx]
        #execute_rate = np.sum(np.array(successes) >= 0.) / cfg.sample

        '''
        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]
        '''

        # Update the best Eureka Output based on Fitness Score (smaller is better)
        if best_fitness < best_fitness_overall:
            best_fitness_overall = best_fitness
            max_reward_code_path = code_paths[best_sample_idx]
        
        best_fitnesses.append(best_fitness)

        #execute_rates.append(execute_rate)
        #max_successes.append(max_success)
        #max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logging.info(f"Iteration {iter}: Best Fitness: {best_fitness}")
        #logging.info(f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}")
        logging.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx].message.content + "\n")
        logging.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")
            
        # Plot the success rate
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle(f'{cfg.env.task}')

        # First subplot: Best Fitness FoM
        x_axis = np.arange(len(best_fitnesses))

        axs[0].plot(x_axis, np.array(best_fitnesses))
        axs[0].set_title("Best Fitness (smaller FoM is better)")
        axs[0].set_xlabel("Iteration")

        # Second subplot: Scatter plot with minimum values highlighted and connected
        for i in range(len(all_fitnesses)):
            # Plot all points for the current iteration
            axs[1].scatter([i] * len(all_fitnesses[i]), all_fitnesses[i], color='blue', alpha=0.5)

            # Highlight the minimum sample for the current iteration
            min_value = np.min(all_fitnesses[i])
            axs[1].scatter(i, min_value, color='red', s=100, edgecolors='black', zorder=5)

            # Connect the minimum values with a line
            if i > 0:
                prev_min_value = np.min(all_fitnesses[i - 1])
                axs[1].plot([i - 1, i], [prev_min_value, min_value], color='red', linestyle='-', linewidth=2)

        axs[1].set_title("Scatter Plot with Highlighted Minimum Values")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Sample Values")
        axs[1].grid(True)

        # Show plot
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        #fig.tight_layout(pad=3.0)
        plt.savefig('summary.png')
        np.savez('summary.npz', best_fitnesses=best_fitnesses, execute_rates=execute_rates, best_code_paths=best_code_paths, max_successes_reward_correlation=max_successes_reward_correlation)



        if len(messages) == 2:
            messages += [{"role": "assistant", "content": responses[best_sample_idx].message.content}]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {"role": "assistant", "content": responses[best_sample_idx].message.content}
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open('messages.json', 'w') as file:
            json.dump(messages, file, indent=4)
    
    # Evaluate the best reward code many times
    if max_reward_code_path is None: 
        logging.info("All iterations of code generation failed, aborting...")
        logging.info("Please double check the output env_iter*_response*.txt files for repeating errors!")
        exit()
    
    logging.info(f"Task: {task}, All Fitness Scores(FoM) {all_fitnesses}")
    logging.info(f"Task: {task}, Best Fitness Score(FoM) {best_fitness_overall}, Best Reward Code Path: {max_reward_code_path}")
    #logging.info(f"Task: {task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}")
    logging.info(f"Evaluating best reward code {cfg.num_eval} times")
    shutil.copy(max_reward_code_path, output_file)
    
    eval_runs = []
    for i in range(cfg.num_eval):
        set_freest_gpu()
        
        # Execute the python file with flags
        rl_filepath = f"reward_code_eval{i}.txt"
        with open(rl_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{PUREJAXRL_ROOT_DIR}/jax_train.py',  
                                        'hydra/output=subprocess',
                                        f'task={task}{suffix}', f'wandb_activate={cfg.use_wandb}',
                                        f'wandb_entity={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                        f'headless={not cfg.capture_video}', f'capture_video={cfg.capture_video}', 'force_render=False', f'seed={i}',
                                        ],
                                        stdout=f, stderr=f)

        block_until_training(rl_filepath)
        eval_runs.append(process)

    reward_code_final_fitnesses = []
    reward_code_final_successes = []
    reward_code_correlations_final = []
    for i, rl_run in enumerate(eval_runs):
        rl_run.communicate()
        rl_filepath = f"reward_code_eval{i}.txt"

        with open(rl_filepath, 'r') as f:
            file_content = f.read()
        # Use a regular expression to extract the values of x0, x1, ..., x15
        pattern = r'x(\d+):\s*([-\d.e+]+)'
        matches = re.findall(pattern, file_content)
        # Convert the matches to a dictionary
        extracted_values = {f'x{match[0]}': float(match[1]) for match in matches}
        spice_script_args = ['python', '-u', f'{HSPICE_ROOT_DIR}/SPICE.py']
        for key, value in extracted_values.items():
            spice_script_args.append(f"--{key}={value}")
        # Adding --env_name argument
        spice_script_args.append(f'--env_name={task}-custom')
        # Execute SPICE.py at HSPICE_ROOT_DIR using x and cfg.
        final_sim_eval_filepath = f"reward_code_eval{i}_sim.txt"
        with open(final_sim_eval_filepath, 'w') as f:
            process = subprocess.Popen(spice_script_args, stdout=f, stderr=f)
        block_until_sim(final_sim_eval_filepath, log_status=True, iter_num=iter, response_id=i)
        with open(final_sim_eval_filepath, 'r') as f:
            sim_str = f.read()
        # Using regular expression to extract the number following "FoM:"
        match = re.search(r"'FoM': ([\d\.]+)", sim_str)
        if match:
            fom_number = float(match.group(1))
            reward_code_final_fitnesses.append(fom_number)
        else:
            fom_number = DUMMY_FAILURE

    logging.info(f"Final Fitness Mean: {np.mean(reward_code_final_fitnesses)}, Std: {np.std(reward_code_final_fitnesses)}, Raw: {reward_code_final_fitnesses}")
    #logging.info(f"Final Success Mean: {np.mean(reward_code_final_successes)}, Std: {np.std(reward_code_final_successes)}, Raw: {reward_code_final_successes}")
    #logging.info(f"Final Correlation Mean: {np.mean(reward_code_correlations_final)}, Std: {np.std(reward_code_correlations_final)}, Raw: {reward_code_correlations_final}")
    #np.savez('final_eval.npz', reward_code_final_successes=reward_code_final_successes, reward_code_correlations_final=reward_code_correlations_final)
    np.savez('final_eval.npz', reward_code_final_fitnesses=reward_code_final_fitnesses)

if __name__ == "__main__":
    main()