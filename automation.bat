@echo off


cd C:\Users\Amzad\Desktop\Dncnn
call conda create -n myenv python=3.6 -y 
call conda activate myenv
call pip install .
call pip install -r requirements.txt
cd C:\Users\Amzad\Desktop\Dncnn\src\dncnn\components
cls