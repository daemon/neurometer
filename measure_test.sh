chrt --rr 1 python2 -m examples.measure_component build_table lenet5_conv1 --ranges '{conv1_out:[1,21]}' --method grid --n_trials 200 --output_file lenet5_conv1-test.csv
