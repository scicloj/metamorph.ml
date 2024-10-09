FROM mcr.microsoft.com/devcontainers/java:11-bullseye

RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/babashka/babashka/master/install)"
RUN bash -c "bash < <(curl -s https://raw.githubusercontent.com/clojure-lsp/clojure-lsp/master/install)"
RUN apt-get update && apt-get install -y python3-pip
RUN pip install mlflow