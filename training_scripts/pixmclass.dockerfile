FROM training_dnn
RUN pip install git+https://github.com/jeanollion/dataset_iterator.git
RUN pip install git+https://github.com/jeanollion/pix_mclass.git
RUN wget https://gist.githubusercontent.com/jeanollion/0f70230391ff79dd8883c97c919e21b8/raw -O training_core.py
RUN wget https://gist.githubusercontent.com/jeanollion/b9b1cf0674858c0e7aaf8bf68bc4b7f5/raw -O train.py
ENTRYPOINT ["/bin/bash"]