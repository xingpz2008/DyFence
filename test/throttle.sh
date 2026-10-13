#!/bin/bash
######
# Network profiles for DyFence benchmarks.
# Based on emp-toolkit/emp-readme/scripts/throttle.sh.
######

## Replace DEV=lo with a physical interface when running across machines.
DEV=lo

if [ "$(id -u)" -eq 0 ]
then
    TC=(tc)
else
    TC=(sudo tc)
fi

if [ "$1" == "del" ]
then
    echo "Cleaning up network rules on $DEV..."
    "${TC[@]}" qdisc del dev $DEV root 2>/dev/null
fi

if [ "$1" == "lan" ]
then
    echo "Setting up QoS-Throttled LAN (400 Mbps, No extra delay)..."
    "${TC[@]}" qdisc del dev $DEV root 2>/dev/null
    
    ## 400 Mbps bandwidth.
    ## The limit and burst values are sized for large ZK message bursts.
    "${TC[@]}" qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 4mbit limit 10mb
fi

if [ "$1" == "wan" ]
then
    echo "Setting up Regional WAN (150 Mbps, 30ms RTT)..."
    "${TC[@]}" qdisc del dev $DEV root 2>/dev/null
    
    ## 150 Mbps bandwidth.
    ## The limit and burst values are sized for large ZK message bursts.
    "${TC[@]}" qdisc add dev $DEV root handle 1: tbf rate 150mbit burst 2mbit limit 5mb

    ## 15 ms one-way delay. On loopback this gives approximately 30 ms RTT.
    "${TC[@]}" qdisc add dev $DEV parent 1:1 handle 10: netem delay 15msec
fi
