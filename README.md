# Optimizer

This repository contains a solution for optimizing a response variable while meeting specifications on other response variables.
The solution uses mathematical and statistical models to analyze limited experimental or simulated data points to achieve the optimal output.

## Results

**The optimal parameters for the minima found were:**

- C: `253.76884422110552`
- R: `46.35678391959799`

**And the minimum cost found was:**

- T: `32.433675076245216`

### Final plot


<div style="text-align: center;">
  ‎ ‎ ‎ ‎ ‎‎ ‎ ‎ ‎ ‎ ‎ ‎  ‎ <img src="https://github.com/user-attachments/assets/fa633e44-5222-4225-ae6d-c7cce4d0bf5e" alt="final-result" height="450" width="720">
</div>


## Repository Structure

- `./sol/`: The final submission data and scripts to run the optimization
- `./requirements.txt`: A list of required Python packages
- `./imgs/`: Contains images of results after multiple stages of optimization
- `./tests/`: Contains approach and initial testing of methods
- `./apk/`: Directory containing `.apk` that provides results for the Black-Box function

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

## Setup Instructions

1. **Clone the repository:**

    ```sh
    git clone https://github.com/sudo-boo/optimizer-azeotropy
    cd optimizer-azeotropy
    ```

2. **Install dependencies:**

    Make sure you have `pip` installed, then run:

    ```sh
    pip install -r requirements.txt
    ```

## Run the Code

To execute the optimization script, run:

```sh
python ./sol/final-solution.py

