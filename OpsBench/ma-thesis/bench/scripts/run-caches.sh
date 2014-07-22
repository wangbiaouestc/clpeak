#! /bin/bash

arch="fermi"
bench_root="/home/michael/Projects/ma-thesis/bench"
bench_res=$bench_root/results

# cache experiments
res=$bench_res/cachememory

$bench_root/bench --cache --log-stride $res/$arch/$arch-auto-nowarm.csv
$bench_root/bench --cache --log-stride --cache-prefer l1 $res/$arch-48kl1-nowarm.csv
$bench_root/bench --cache --log-stride --cache-prefer shared $res/$arch-16kl1-nowarm.csv

location=$(pwd)
cd $bench_root/bench
scons disablel1=true
cd $location
$bench_root/bench --cache --log-stride $res/$arch-nol1-nowarm.csv

# shared memory experiments
res=$bench_res/shared

$bench_root/bench --shared --log-stride $res/$arch/$arch-shared.csv
