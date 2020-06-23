#!/bin/bash
fileid="1sTfUoNPid9zG_MgV--lWZTBP1XZpmcK8"
filename="IQON3000.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}