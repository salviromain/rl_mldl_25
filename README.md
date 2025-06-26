# Reinforcement Learning Project (Course Project – MLDL 2025, Politecnico di Torino)

### General Description

This repository, originally forked from [rl_mldl_25](https://github.com/lambdavi/rl_mldl_25), contains all the code developed by Group 4 (Salvi, Sernia, Vittimberga, Zoccatelli) for Tasks 2–6 of the *Machine Learning and Deep Learning* course, Master's program in Data Science and Engineering at Politecnico di Torino.

To view the `Report` with hyperlinks working correctly, download at this [link](https://github.com/salviromain/rl_mldl_25/raw/main/Report.pdf)

## Repository Structure and Branches

Three branches were created from the `main` branch, each corresponding to specific tasks:

- `REINFORCE-PPO`: Contains code for **Tasks 2** and **4**.  
- `ActorCritic`: Contains code for **Task 3**.  
- `PPO-UDR`: Contains code for **Task 6**.

The models referenced in `Report.pdf` are located in their respective branches.

## Model Testing Instructions

> **Note**: Prior installation of all required dependencies is assumed.

To reproduce the results shown in `Report.pdf`, use the following commands:

- **REINFORCE model** (in the `REINFORCE-PPO` branch):

  ```bash
  python test.py --model <path_to_model_mdl> --episodes 500
  ```

- **Actor-Critic model** (in the `ActorCritic` branch):

  ```bash
  python test.py --model <path_to_model_mdl> --episodes 500
  ```

- **PPO with UDR** (in the `PPO-UDR` branch):

  ```bash
  python testppo.py --model <path_to_model_zip> --episodes 500 --render True
  ```
