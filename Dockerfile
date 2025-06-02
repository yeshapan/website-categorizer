#start with an official Python runtime as a parent image
FROM python:3.12-slim

#set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION=1.8.3 # Or your specific Poetry version
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false # Install dependencies in the system site-packages

#install Poetry
RUN apt-get update && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="${POETRY_HOME}/bin:${PATH}"

#set the working directory in the container
WORKDIR /app

#copy only files necessary for dependency installation first to leverage Docker cache
COPY pyproject.toml poetry.lock ./

#install project dependencies using Poetry
# --no-dev: Do not install development dependencies
# --no-interaction: Do not ask any interactive questions
# --no-ansi: Disable ANSI output
RUN poetry install --no-dev --no-interaction --no-ansi

#NLTK data download (stopwords) - run as part of the build
RUN python -m nltk.downloader stopwords

#copy the rest of the application code into the container
COPY . .

#make port 8501 available to the world outside this container (Streamlit default port)
EXPOSE 8501

#command to run the Streamlit application
#use `app/app.py` as Streamlit Cloud usually expects the app at the root or a specified path.
#here, we are running it from the WORKDIR /app.
#the Streamlit server will listen on all available network interfaces (0.0.0.0).
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]