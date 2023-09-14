FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
# get private key from gist. important: must end by a new line
RUN wget https://gist.githubusercontent.com/sabilab/bc6a3acdd43472b6dc2a22b6244c0d49/raw/bacmman-distnet2d-read -O id25519
RUN chmod 400 id25519
ENV GIT_SSH_COMMAND='ssh -i /id25519 -oStrictHostKeyChecking=no'
RUN pip install git+ssh://git@github.com/jeanollion/distnet2d.git
COPY training_core.py /training_core.py
COPY training_distnet2d.py /train.py
ENTRYPOINT ["/bin/bash"]