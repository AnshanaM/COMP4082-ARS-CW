
# COMP4082-ARS-CW

Ensure all packages are installed using package manager [pip](https://pip.pypa.io/en/stable/). For example:

```bash

pip install  gymnasium

```

## Testing a model

Models are stored in the *results* folder in each algorithm directory.

To test a model using the *.pt* file, navigate to the appropriate algorithm folder and open the *test.py* file. Alter the environment ID `env_id` and add the model path like so:

```bash

env_id =  "CartPole-v1"

model_path =  'test/cartpole.pt'

```

Save, use the terminal and navigate to the algorithm directory and run using:

```bash

python test.py

```

## Training the algorithms

The training session is of 5 independent runs in 30K time frames.

### 1. NROWAN-DQN original algorithm (Pendulum and CartPole)

The directory `nrowan_dqn` contains all the scripts required to run the NROWAN-DQN algorithm on the Pendulum environment. CartPole environment was trained previously using the same algorithm and results are saved in the `results` directory.

### 2. NROWAN-DQN with action noise (Pendulum)

Run the following in the terminal of the directory of `nrowan_dqn pendulum_action_noise`:

```bash

python agent.py

```

### 3. NROWAN-DQN with action noise (CartPole)

If training **with** PER, run the following in the terminal of the directory of `nrowan_dqn_cartpole_action_noise`:

```bash

python agent_prioritised_replay.py

```

If training **without** PER, run the following in the terminal of the directory of `nrowan_dqn_cartpole_action_noise:

```bash

python agent.py

```

### 4. Diverse Priors with NROWAN-DQN (CartPole)
This algorithm is a replica of the BSPD algorithm proposed in the Diverse Priors in Reinforcement Learning Paper but it involves an ensemble of NROWAN-DQNs instead of standard DQNs with a combined loss function.
To run the training session, run the following in the directory of `diverse_priors_nrowan`:
```bash
python BSDP.py
```