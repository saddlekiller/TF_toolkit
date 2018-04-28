# python demo.py basic_lstm basic_lstm
#python demo.py pure_affine pure_affine
#python demo.py normalized_affine normalized_affine
#python demo.py partial_normalized_affine partial_normalized_affine
# python demo.py normalized_rnn normalized_rnn
# python demo.py basic_rnn basic_rnn
# python demo.py normalized_rnn2 normalized_rnn2


python model.py BasicRNNModel BasicRNNCell
python model.py BasicRNNModel NormalizedRNNCell
python model.py BasicRNNModel LSTMCell
python model.py BasicRNNModel GRUCell

python model.py NormRNNModel BasicRNNCell
python model.py NormRNNModel NormalizedRNNCell
python model.py NormRNNModel LSTMCell
python model.py NormRNNModel GRUCell
