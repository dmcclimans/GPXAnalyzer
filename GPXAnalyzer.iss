; Installer script for GPXAnalyzer application.
;
#define MyAppName "GPXAnalyzer"
#define MyAppExeName "GPXAnalyzer.exe"
#define MyAppPublisher "Don McClimans"
#define MyAppCopyrightYear "2021"

#define InstallerNameBase "GPXAnalyzer"
#define OutputDirectory "dist"
#define InputDirectory "dist\gpxanalyzer"

// Create the output file name: something like: GPXAnalyzer_setup_1_2_3
// Also set the version of the setup program to match the app.
#define Major
#define Minor
#define Rev
#define Build
#define MyAppVersion GetStringFileInfo(InputDirectory + "\" + MyAppExeName, FILE_VERSION)
#expr ParseVersion(InputDirectory + "\" + MyAppExeName, Major, Minor, Rev, Build)
#if (0 == Build)
    #define InstallerVersionString = str(Major) + "_" + str(Minor) + "_" + str(Rev)
#else
    #define InstallerVersionString = str(Major) + "_" + str(Minor) + "_" + str(Rev) + "_" + str(Build)
#endif

#define InstallerName = InstallerNameBase +	"_setup_" + InstallerVersionString

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{4E631156-DC52-4580-B865-AD8D5DE8BCB6}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
VersionInfoVersion={#MyAppVersion}
VersionInfoCopyright=Copyright © {#MyAppCopyrightYear} {#MyAppPublisher}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName="{#MyAppName}"
OutputDir={#OutputDirectory}
OutputBaseFilename={#InstallerName}
Compression=lzma
SolidCompression=yes
; Only install on 64-bit Windows 10
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
MinVersion=0,10.0
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}";

[Files]
; NOTE: Don't use "Flags: ignoreversion" on any shared system files
Source: "{#InputDirectory}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
