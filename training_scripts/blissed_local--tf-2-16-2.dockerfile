FROM training_dnn:tf-2.16.2
RUN pip install tf-keras~=2.16
RUN pip install --upgrade h5py==3.11.0
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY ssnb_denoising /ssnb_denoising
RUN pip install /ssnb_denoising
COPY training_scripts/training_scripts/training_core.py /training_core.py
COPY training_scripts/training_scripts/training_blissed.py /train.py
RUN chmod a+r /training_core.py
RUN chmod a+r /train.py
ENV NUMBA_NUM_THREADS=1
ENTRYPOINT ["/bin/bash"]