pyinstaller --onedir --noconsole --version-file=GPXAnalyzerVersion.txt ^
 --noupx  --clean --noconfirm ^
 --log-level=WARN ^
 --add-binary AppSplashscreen.png;. ^
 --add-binary AppScreenshot1.png;.  --add-binary AppChart1.png;. ^
 --add-data Readme.md;. --add-data License.txt;. ^
 gpxanalyzer.py
@pause
