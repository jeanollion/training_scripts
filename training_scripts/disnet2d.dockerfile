FROM jeanollion/training_dnn:tf-2.7.1
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
RUN wget https://gist.githubusercontent.com/jeanollion/4064dc66a68cb99e87ed751fcdba0f75/raw -O id25519
RUN chmod 400 id25519
ENV GIT_SSH_COMMAND='ssh -i /id25519 -oStrictHostKeyChecking=no'
RUN pip install git+ssh://git@github.com/jeanollion/distnet2d.git
RUN wget https://gist.githubusercontent.com/jeanollion/4aea9bef9c4b98aa5f8084e1be5ed6ee/raw -O training_core.py


ENTRYPOINT ["/bin/bash"]