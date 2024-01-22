call activate jefapato
RMDIR /S /Q build
RMDIR /S /Q dist
pyinstaller --console --onefile --name JeFaPaTo --add-data frontend:frontend --add-data jefapato:jefapato --add-data examples:examples --icon "frontend\assets\icons\icon.ico" main.py
CD dist
RENAME JeFaPaTo.exe JeFaPaTo_windows.exe