FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
# get private key from gist. important: must end by a new line
RUN wget https://gist.githubusercontent.com/sabilab/bc6a3acdd43472b6dc2a22b6244c0d49/raw -O id25519
RUN chmod 400 id25519
ENV GIT_SSH_COMMAND='ssh -i /id25519 -oStrictHostKeyChecking=no'
RUN pip install git+ssh://git@github.com/jeanollion/distnet2d.git
RUN wget https://gist.githubusercontent.com/jeanollion/4aea9bef9c4b98aa5f8084e1be5ed6ee/raw/training_core.py -O training_core.py
RUN wget https://gist.githubusercontent.com/jeanollion/789b9dbbda92da548401c7250f1631ad/raw/training_distnet2d.py -O train.py
ENTRYPOINT ["/bin/bash"]