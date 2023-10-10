
for network in 2by2 3by3 4by4 5by5 hangzhou manhattan manhattan-large; do
    for scheme in comb timing; do
        python meta_train.py --network $network --scheme $scheme --model anp --exp_id trial1 &
        wait
    done
done
