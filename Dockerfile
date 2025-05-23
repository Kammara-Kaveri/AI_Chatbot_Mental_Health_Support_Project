# Use an official Python runtime as the base image
   FROM python:3.11-slim

   # Install Java (required for language_tool_python)
   RUN apt-get update && apt-get install -y default-jre && rm -rf /var/lib/apt/lists/*

   # Set the working directory
   WORKDIR /app

   # Copy the requirements file and install dependencies
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy the rest of the application code
   COPY . .

   # Expose the port your app runs on (Render requires this)
   EXPOSE 10000

   # Command to run the app with Gunicorn
   CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]