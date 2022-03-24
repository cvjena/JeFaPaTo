echo CREATE ANACONDA ENVIRONMENT

call %UserProfile%\Anaconda3\Scripts\activate.bat base
conda create -n jefapato python=3.9
conda activate jefapato
pip install -r requirements.txt
pause > nul
