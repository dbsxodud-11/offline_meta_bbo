
for network in 2by2 3by3 4by4 5by5 hangzhou manhattan manhattan-large; do
    for scheme in comb timing; do
        for scenario in {0..9}; do
            python meta_test.py --network $network --scheme $scheme --model anp --exp_id trial1 --scenario_id $scenario_id &
            wait
        done
    done
done
