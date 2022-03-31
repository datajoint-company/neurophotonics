FROM datajoint/djbase:py3.9-debian-fcd8909

USER root
RUN apt update && \
    apt-get install -y ssh git

WORKDIR $HOME

# Copy user's local fork of elements and workflow
COPY ./neurophotonics $HOME/neurophotonics

# Install the workflow
RUN pip install -e $HOME/neurophotonics

WORKDIR $HOME