.PHONY: build run-parallel all clean

# Define the file that contains the commands
COMMANDS_FILE = commands.txt

# --- Targets ---

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker compose build trainer
	@echo "Docker image built."

# Run the commands from $(COMMANDS_FILE) in parallel using Docker containers
# Each line in $(COMMANDS_FILE) should be a complete command.
# Adjust -P <number> to control the level of parallelism (e.g., -P 4 for 4 jobs)
run-parallel: build
	@echo "Running commands in parallel from $(COMMANDS_FILE)..."
	@if [ ! -s $(COMMANDS_FILE) ]; then \
		echo "Error: $(COMMANDS_FILE) not found or is empty."; \
		exit 1; \
	fi
	# Use xargs to run commands from the file in parallel
	# -I {} : Replace {} with each line read from the input
	# -P <num> : Run <num> jobs in parallel (adjust the number as needed, e.g., based on CPU cores)
	# --rm : Automatically remove the container after it exits
	# ADD THE -T FLAG HERE
	cat $(COMMANDS_FILE) | xargs -I {} -P 4 docker compose run --rm -T trainer {}
	@echo "Parallel execution finished."

# Default target: Build and run parallel
all: build run-parallel

# Clean up generated files (in this case, just the commands file if desired, though we keep it)
# clean:
# 	@echo "Cleaning up generated files..."
# 	@echo "No files to clean in this setup unless you want to remove commands.txt"
# 	# rm -f $(COMMANDS_FILE)
# 	@echo "Cleanup complete."
