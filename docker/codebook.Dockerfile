ARG JHUB_VER
ARG PY_VER
ARG DIST
FROM datajoint/djlabhub:${JHUB_VER}-py${PY_VER}-${DIST}

ARG REPO_OWNER
ARG REPO_NAME
WORKDIR /tmp
RUN git clone http://github.com/${REPO_OWNER}/${REPO_NAME} && \
    pip install ./${REPO_NAME} && \
    cp -r ./${REPO_NAME}/ /home/notebooks/
WORKDIR /home/notebooks