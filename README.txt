This is the final deliverable for "Team 17", consisting of:

- Marko Caklovic (2349514C)
- Igor Drozhilkin (2353454D)
- Apoorva Tamaskar (2349061T)

The package consists of the following files:

final_ai_report.pdf - our final report for this assessment

myagents.py - all code needed to run the Random/Simple/Realistic Agents in Malmo

q_learning_optimize.py - supplemental code which shows how we optimized the alpha/gamma/epsilon parameters for Q learning. It is not necessary to run this code to get myagents.py to work. It's merely included for demonstrative purposes, since it is referenced in the report. We recommend you run this script with Pypy, as it is performance intensive. 

grids - a folder which contains input data for q_learning_optimize.py.


To run the Random agent, execute:
python myagents.py -a Random

To run the Simple agent, execute:
python myagents.py -a Simple

To run and train the Realistic agent, execute:
python myagents.py -a Realistic -n 30 --training 1
(this will run and train the Realistic agent across 30 iterations of the same scenario. Training data is saved to a folder called 'q_tables_dir')

After training, to run the trained Realistic agent, executed:
python myagents.py -a Realistic -n 5 --training 0
(this will run the trained agent 5 times)
