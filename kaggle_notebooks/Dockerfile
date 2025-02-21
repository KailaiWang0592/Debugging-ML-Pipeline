FROM continuumio/miniconda3  

# Copy the Conda environment file into the container
COPY environment.yml /tmp/environment.yml

# Create the Conda environment
RUN conda env create -f /tmp/environment.yml

# Activate the Conda environment
RUN echo "conda activate kaggle_mega" >> ~/.bashrc
ENV PATH /opt/conda/envs/kaggle_mega/bin:$PATH

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
