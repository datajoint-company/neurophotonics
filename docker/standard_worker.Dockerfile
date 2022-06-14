ARG mode=djbase
FROM datajoint/${mode}

RUN \
  echo git >> /tmp/apt_requirements.txt && \
  /entrypoint.sh echo complete && \
  rm /tmp/apt_requirements.txt

# Copy user's local fork of elements and workflow
COPY --chown=anaconda:anaconda ./setup.py ./requirements.txt ./README.md ./
COPY --chown=anaconda:anaconda ./neurophotonics ./neurophotonics

# Install the workflow
RUN \
  umask u=rwx,g=rwx,o-rwx && \
  mkdir -p /main/datajoint/blob && \
  pip install -e .
