#!/bin/bash

pip install gdown
if [ $? -eq 0 ]; then
	gdown https://drive.google.com/uc?id=1835io6kDGkUPvJHdA28lL_QoRlaqcciq
fi	
