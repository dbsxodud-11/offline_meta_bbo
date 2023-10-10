
for network in 2by2 3by3 4by4 5by5 hangzhou manhattan manhattan-large; do
    for scheme in comb timing; do
        python collect_data.py --network $network --scheme $scheme &
        wait
    done
done
