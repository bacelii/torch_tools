# Use the official PyTorch image with CUDA support
FROM python:3.9-slim


# Install Jupyter and common Python libraries
RUN pip install --no-cache-dir \
    notebook \
    jupyterlab \
    ipykernel \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    torchvision \
    torch \
    torchaudio

# Expose Jupyter Notebook port
EXPOSE 8888

# Set the working directory
WORKDIR /Applications

# Command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
