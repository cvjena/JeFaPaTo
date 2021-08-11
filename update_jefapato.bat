echo UPDATE ANACONDA ENVIRONMENT

call %UserProfile%\Anaconda3\Scripts\activate.bat jefapato
conda env update -f env.yml
pause > nul