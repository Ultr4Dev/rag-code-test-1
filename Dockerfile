# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables for the OpenAI API key and port
ENV OPENAI_API_KEY your_openai_api_key
ENV PORT 81

# Create and set the working directory in the container
WORKDIR /app

# Copy the application files into the container
COPY . /app

RUN mkdir static
# Install required Python packages
RUN pip install -r requirements.txt

# Expose the port your FastAPI application will run on
EXPOSE $PORT

# Command to run the FastAPI application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "81"]
