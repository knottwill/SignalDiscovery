# Statistics 1 - Coursework

This project is primarily concerned with the use of statistical methods to discover guassian signals amoung an exponentially decaying background (reminiscent of the Higgs boson discovery). We perform Neyman-Pearson hypothesis tests for the presence of signal by fitting probability distributions to data using maximum likelihood estimation, and utilising the Neyman-Pearson test statistic. Read the assigment in `assignment.pdf` and the report in `report/main.pdf` for further details. 

The codebase & report contained in this repository was submitted in fulfillment of the coursework assigment detailed in `assigment.pdf`. This submission recieved a grade of 89% (may be subject to moderation in Final Examiners' Meeting in September 2024).

### Usage

Clone the repository:

```bash
$ git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/s1_assessment/wdk24.git
$ cd wdk24
```

To generate the docker image and run the container, navigate to the root directory and use:

```bash
$ docker build -t <image_name> .
$ docker run -ti <image_name>
```

To replicate the results and/or plots from parts c, d, e, f and g, run the following scripts:

```bash
$ python src/solve_part_c.py
$ python src/solve_part_d.py
$ python src/solve_part_e.py
$ python src/solve_part_f.py
$ python src/solve_part_g.py
```

Results will be printed to terminal and plots will be saved in `plots/`

### Timing

I ran all scripts on my personal laptop with the following specifications:
- Chip:	Apple M1 Pro
- Total Number of Cores: 8 (6 performance and 2 efficiency)
- Memory (RAM): 16 GB
- Operating System: macOS Sonoma v14.0

The scripts `solve_part_c.py`, `solve_part_d.py`, `solve_part_e.py` all ran in less than 10 seconds.

The script `solve_part_f.py` took roughly 15 minutes to run.

The script `solve_part_g.py` took roughly 50 minutes to run. 

### Project Structure

All code is contained inside the `src/` folder. The `solve_part_*` scripts are those which can be used to replicate the results in the report. The other modules in the `src/` folder contain functionality to assist the scripts. 

- `NP_analysis.py` - Contains functions to assist with the analysis of the 'probability of discovery vs dataset size' data in parts (f) and (g)
- `discovery.py` - Contains function to calculate the probability of discovery for a given dataset size. 
- `distributions.py` - Contains all probability density functions and cumulative density functions relevant to the project.
- `generation.py` - Contains functionality to generate datasets (from the 'signal plus background' model and the 'two signals plus background' model).
- `hypothesis_test.py` - Contains functionality to perform hypothesis tests on datasets.
