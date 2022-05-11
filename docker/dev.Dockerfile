FROM datajoint/djbase:py3.9-debian

RUN \
  echo git >> /tmp/apt_requirements.txt && \
  /entrypoint.sh echo complete && \
  rm /tmp/apt_requirements.txt

# Copy user's local fork of elements and workflow
COPY --chown=anaconda:anaconda ./setup.py ./requirements.txt ./README.md dj_local_conf.json ./
COPY --chown=anaconda:anaconda scripts ./scripts
COPY --chown=anaconda:anaconda ./neurophotonics ./neurophotonics
COPY --chown=anaconda:anaconda ./workdir ./workdir

# Install the workflow
RUN \
	umask u=rwx,g=rwx,o-rwx && \
	mkdir /main/data_store && \
	pip install -e .
