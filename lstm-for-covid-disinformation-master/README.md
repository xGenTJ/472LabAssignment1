# LSTM for Covid disinformation

Simple LSTM classification model for tweets containing misinformation about Covid-19.

In order to run the training script `main.py`, clone the repository, install the requirements, download the appropriate [embeddings model](http://vectors.nlpl.eu/repository/20/6.zip), and put the file in the same folder. Don't forget to unzip the model.

Using Python 3.7.9, you can install the requirements using `pip`. Windows users must install PyTorch separately using  the following command:

> ```$ pip install torch===1.7.0 -f https://download.pytorch.org/whl/torch_stable.html```

Windows users must also make sure to have a Visual C++ distribution. You can find one [here](https://aka.ms/vs/16/release/vc_redist.x64.exe).

Once done, installing the remaining requirements is done by entering

> ```$ pip install -r requirements.txt ```


## Important note on training

Because of the nature of the random initialisation of the model parameters, you are encouraged to train the LSTM model more than once and compare the difference in performance each time.