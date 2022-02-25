echo CREATE ANACONDA ENVIRONMENT

call %UserProfile%\Anaconda3\Scripts\activate.bat base
conda env create -f env.yml
pause > nul
