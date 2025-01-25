FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies from requirements.txt using pip
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the project files into the container
COPY . .
