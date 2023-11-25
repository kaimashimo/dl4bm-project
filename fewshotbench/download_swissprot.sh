#!/bin/bash

pip install gdown
if [ $? -eq 0 ]; then
	gdown https://drive.google.com/uc?id=1a3IFmUMUXBH8trx_VWKZEGteRiotOkZS
fi	
