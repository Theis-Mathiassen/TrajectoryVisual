# Use a lightweight Python base image
FROM python:3.9-slim

# Install necessary system dependencies for pip and potentially uv build (minimal)
# We need curl for the original install method, but it's good to keep apt-get update/clean
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv using pip
# Although the official docs recommend the standalone installer or pipx,
# this fulfills the request to install via pip.
RUN pip install --no-cache-dir uv

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file (assuming you use pyproject.toml or requirements.txt)
# If using uv with pyproject.toml/requirements.txt, the next steps install dependencies
COPY pyproject.toml uv.lock ./

# Use uv to install dependencies
RUN uv sync --locked

# If using pip with requirements.txt, uncomment the next two lines (and remove uv steps above)
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . /app

# Command to run the script (will be overridden by docker compose run/exec)
# CMD ["uv", "run", "main.py"]
