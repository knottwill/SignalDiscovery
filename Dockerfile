# use the Miniconda3 base image
FROM continuumio/miniconda3

# install git and curl
# (final line removes package lists which are no longer needed)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# set the working directory in the Docker image
WORKDIR /stats

# copy the current directory contents into the container at /sudoku_master
COPY . /stats

# update the conda environment
RUN conda env update --file environment.yml --name base
