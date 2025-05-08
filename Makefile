.PHONY: build run-parallel all clean

# Define the file that contains the commands
COMMANDS_FILE = commands.txt

# Define the base directory for storing job results on the host
RESULTS_BASE_DIR = results

# --- Targets ---

# Build the Docker image
build:
	@echo "Building Docker image..."
	docker compose build trainer
	@echo "Docker image built."

# Run the commands from $(COMMANDS_FILE) in parallel using Docker containers
# Each line in $(COMMANDS_FILE) should be a complete command.
# Each command will run in a container with a unique output directory mounted.
# Adjust -P <number> to control the level of parallelism (e.g., -P 4 for 4 jobs)
run-parallel: build
	@echo "Running commands in parallel from $(COMMANDS_FILE)..."
	@if [ ! -s $(COMMANDS_FILE) ]; then \
		echo "Error: $(COMMANDS_FILE) not found or is empty."; \
		exit 1; \
	fi
	# Ensure the base results directory exists on the host
	@mkdir -p $(RESULTS_BASE_DIR)

	# Use xargs to run a small shell script for each command line
	# The shell script creates a unique directory and runs the container with a dynamic volume mount
	# -I {} : Replace {} with each line read from the input
	# -P <num> : Run <num> jobs in parallel (adjust the number as needed)
	# bash -c "..." : Execute a shell script for each item from xargs using double quotes
	cat $(COMMANDS_FILE) | xargs -I {} -P 4 bash -c " \
		COMMAND=\"{}\"; \
		# Generate a unique ID for the job based on the command string
		# Using MD5 hash of the command for a unique, directory-safe name
		JOB_ID=$$(echo \"$$COMMAND\" | md5sum | cut -d\\\" \\\" -f1); \
		# Define the unique output directory path on the HOST machine
		OUTPUT_DIR=\"$(RESULTS_BASE_DIR)/$$JOB_ID\"; \
		# Create the unique directory on the HOST machine
		mkdir -p \"$$OUTPUT_DIR\"; \
		echo \"--> Starting job $$JOB_ID in $$OUTPUT_DIR: $$COMMAND\"; \
		# Run the container, mounting the unique host directory to the container's working directory (/app)
		# -T : Disable TTY allocation (fixes the previous error)
		# -v \"$$OUTPUT_DIR\":/app : Mount the unique HOST directory as the container's working directory
		# trainer : The service name
		# $$COMMAND : The actual command to run inside the container
		docker compose run --rm -T \
					-v \"$$OUTPUT_DIR\":$(CONTAINER_OUTPUT_PATH) \
					-e JOB_OUTPUT_DIR=$(CONTAINER_OUTPUT_PATH) \
					trainer $$COMMAND; \
		EXIT_CODE=$$?; \
		if [ $$EXIT_CODE -ne 0 ]; then \
			echo \"--> Job $$JOB_ID failed with exit code $$EXIT_CODE\"; \
		fi; \
		# Optional: Add a marker file indicating job completion/status
		# echo \"Job finished\" > \"$$OUTPUT_DIR\"/job_status.txt \
	"
	@echo "Parallel execution finished."
# Default target: Build and run parallel
all: build run-parallel

# Clean up generated files (commands.txt is kept as it's source)
# Option to clean up the results directories if needed:
# clean:
# 	@echo "Cleaning up generated result directories..."
# 	rm -rf $(RESULTS_BASE_DIR)/*
# 	@echo "Cleanup complete."
