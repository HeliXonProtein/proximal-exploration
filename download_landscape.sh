#!/bin/bash

FILE="landscape_params.zip"
ID="1QvpHi-P2-hw2xMwZrpKvF-95kp4SaZWS"
URL="https://docs.google.com/uc?export=download&id=$ID"
COOKIES="./cookies.txt"

wget --load-cookies $COOKIES "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies $COOKIES --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$ID" -O $FILE && rm -rf $COOKIES
unzip $FILE
rm $FILE
