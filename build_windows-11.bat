call activate jefapato
RMDIR /S /Q build
RMDIR /S /Q dist
pyinstaller --console --onefile --name JeFaPaTo --add-data src/jefapato:jefapato --add-data frontend:frontend --add-data examples:examples --icon "frontend\assets\icons\icon.ico" --collect-all=numba --collect-all=stumpy main.py
CD dist
RENAME JeFaPaTo.exe JeFaPaTo_windows.exe