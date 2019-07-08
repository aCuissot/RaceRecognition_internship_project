from dataset_tools import load_for_pred

mobile_model_multitask.load_weights(ckpntlist[0])
        
data, Y_true = load_for_pred(val_dataset, batch_size, shape)
print(Y_true.shape)
Y_pred = mobile_model_multitask.predict(data, batch_size, verbose=1)
print(Y_pred.shape)
print(Y_true.shape)
y_pred = np.argmax(Y_pred, axis=1)
y_true = np.argmax(Y_true, axis=1)
print(y_pred.shape)
print(y_true.shape)
print('Confusion Matrix')
# predneg predpos
#  [[4178  659]  < True negative
#   [ 975 3861]] < True positive
from sklearn.metrics import classification_report, confusion_matrix
#y_pred = [1]*y_true.shape[0] # Per provare, dovrebbe dare recall=1
conf = confusion_matrix(y_true, y_pred, [0, 1, 2, 3])
print(conf)