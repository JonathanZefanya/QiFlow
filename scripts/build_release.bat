@echo off
set PY=py -3.12
%PY% -m pip install -r requirements.txt
%PY% -m pip install pyinstaller
%PY% -m pyinstaller --noconsole --onefile --add-data "assets;assets" --add-data "config;config" main.py
if exist release rmdir /s /q release
mkdir release
copy /y dist\main.exe release\QiFlow.exe
xcopy /e /i /y config release\config
xcopy /e /i /y assets release\assets
