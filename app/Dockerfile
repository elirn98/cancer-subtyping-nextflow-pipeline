# Base image
FROM python:3.12.6

FROM openjdk:11-jre  

RUN apt-get update && apt-get install -y python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required tools
# RUN apt-get update && apt-get install -y curl \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install Nextflow
# RUN curl -s https://get.nextflow.io | bash \
# && chmod +x nextflow \
# && mv nextflow /usr/local/bin/

# # Ensure the binary is available
# RUN which nextflow

# Set working directory
WORKDIR /app

# Copy application files to the container
COPY . /app

# Copy pipeline files into the image
# COPY pipeline.nf nextflow.config /app/

# Default entrypoint for Nextflow
# ENTRYPOINT ["/bin/bash", "-c"]

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose the application port
EXPOSE 5000

# Command to run the application
# CMD ["run", "pipeline.nf"]
