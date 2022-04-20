FROM datajoint/djbase:py3.9-debian

# Copy user's local fork of elements and workflow
COPY ./setup.py ./requirements.txt ./README.md dj_local_conf.json ./
COPY scripts ./scripts
COPY ./neurophotonics ./neurophotonics
COPY ./workdir ./workdir

# Install the workflow
RUN \
	umask u=rwx,g=rwx,o-rwx && \
	mkdir /main/data_store && \
	pip install -e .
