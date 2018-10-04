chrt --rr 1 taskset -c 2 python2 -m examples.measure_component build_table lenet5_fc1 --ranges '{lin1_out:[1,501],conv2_out:[1,51]}' --method grid --n_trials 200 --output_file lenet5_fc1.csv
chrt --rr 1 taskset -c 2 python2 -m examples.measure_component build_table lenet5_conv1 --ranges '{conv1_out:[1,21]}' --method grid --n_trials 200 --output_file lenet5_conv1.csv
chrt --rr 1 taskset -c 2 python2 -m examples.measure_component build_table lenet5_conv2 --ranges '{conv1_out:[1,21],conv2_out:[1,51]}' --method grid --n_trials 200 --output_file lenet5_conv2.csv
