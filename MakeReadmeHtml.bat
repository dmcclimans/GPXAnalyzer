pandoc --standalone -f gfm -M document-css=false^
 --include-in-header %AppData%\Pandoc\PandocSansSerif.css^
 --metadata title="Merge Tempo Disc sensor data into GPX files"^
 -o "Readme.html"^
 "Readme.md"
@pause
