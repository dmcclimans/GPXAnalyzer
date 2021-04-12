pyinstaller --onedir --noconsole --version-file=GPXAnalyzerVersion.txt ^
 --noupx  --clean --noconfirm  --log-level=WARN ^
 --add-binary splashscreen.png;. --add-binary screenshot1.png;. ^
 --add-data Readme.md;. --add-data License.txt;. ^
 gpxanalyzer.py
@pause
