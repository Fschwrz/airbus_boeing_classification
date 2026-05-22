# airbus_boeing_classification
This is a neural network project classifying photos of airplanes by Boeing and Airbus, trained on the ImageNet 2013 FGVC challenge dataset

Due to the file size limit of github, the models need to be downloaded at the following links:
https://www.dropbox.com/scl/fi/jvabada64gf4wg8gml0yn/enginetype_model_3_dense_layers_25_epochs.keras?rlkey=63gtpfj5syefwtra7lo89s16z&st=5j63wcx3&dl=0
https://www.dropbox.com/scl/fi/pgy9kwzdps2h2f17tf26u/quad_model.keras?rlkey=pdr3st6ij58rejq5z7elbdg52&st=v6bxpgfs&dl=0
https://www.dropbox.com/scl/fi/azhvpaajau4l9oc7has6h/rear_model.keras?rlkey=rf57flt8cyw15pkwffzeu7riy&st=poom2tep&dl=0
https://www.dropbox.com/scl/fi/pi42xk5c6n5bw58ns8dzl/twin_model.keras?rlkey=120j3g0yl7x1nig67dqqau32w&st=ooha1ovh&dl=0

For usage, put evaluate_model.py in the same directory as the models and change the FILE_PATH parameter in evaluate_model.py to the file path of the image you want to classify
The jupyter notebook is what I used to train the model, use at your own risk, it may not be easy to read.
