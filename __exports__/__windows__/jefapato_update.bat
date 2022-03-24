echo UPDATE ANACONDA ENVIRONMENT

call %UserProfile%\Anaconda3\Scripts\activate.bat jefapato
pip install -r requirements.txt
pause > nul
