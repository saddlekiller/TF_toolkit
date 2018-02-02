import json

json_data = json.load(open('../models/mnist_demo/model.conf', 'r'))
layers_config = json_data['layers']
# print(json_data)
# print(layers_config)

pairs = []
for key, item in layers_config.items():
    pairs.append((item['inputs'], key))
    # pairs.append((key, item['outputs']))

print(pairs)
print('=============================')
print(list(set(pairs)))
print('=============================')
print(len(pairs))
print(len(list(set(pairs))))
print((1,2)==(2,1))
